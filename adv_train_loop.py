import os
import copy
from utils import create_summary_writer
import torch
import ignite
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import (Engine, Events)
from ignite.handlers import (ModelCheckpoint, EarlyStopping,
                             Checkpoint, DiskSaver, global_step_from_engine)
from ignite.metrics import Accuracy, Loss
import torch.nn.functional as F
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, CarliniLInfMethod, DeepFool
from art.estimators.classification import PyTorchClassifier


def adv_train_loop(model, params, ds, min_y, base_data, model_id, attack_type, device, batch_size, max_epochs=5):
    print('training adversarial:', attack_type)
    ds_train, ds_valid = ds
    min_y_train, min_y_val = min_y
    original_model = copy.deepcopy(model)  # used to generate adv images for the trained model
    original_model.eval()
    model = copy.deepcopy(model)  # making a copy so that original model is not changed
    model_id = f'{model_id}_{attack_type}'

    with create_summary_writer(model, ds_train, base_data, model_id, device=device) as writer:
        lr = params['lr']
        mom = params['momentum']
        wd = params['l2_wd']
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=mom, weight_decay=wd)
        sched = ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        funcs = {'accuracy': Accuracy(),
                 'loss': Loss(F.cross_entropy)}
        loss = funcs['loss']._loss_fn

        acc_metric = Accuracy(device=device)
        loss_metric = Loss(F.cross_entropy, device=device)

        acc_val_metric = Accuracy(device=device)
        loss_val_metric = Loss(F.cross_entropy, device=device)

        classifier = PyTorchClassifier(
            model=original_model,
            clip_values=(0, 1),
            loss=nn.CrossEntropyLoss(),
            optimizer=optimizer,
            input_shape=(3, 64, 64),
            nb_classes=200,
        )

        attack = None

        if attack_type == "fgsm":
            attack = FastGradientMethod(estimator=classifier, eps=0.2)
        elif attack_type == "bim":
            attack = BasicIterativeMethod(estimator=classifier, eps=0.2)
        elif attack_type == "carlini":
            attack = CarliniLInfMethod(classifier=classifier)
        elif attack_type == "deepfool":
            attack = DeepFool(classifier=classifier)


        def train_step(engine, batch):
            model.train()
            x, y = batch
            x = x.to(device)
            x_adv = torch.Tensor(attack.generate(x=x.cpu())).to(device)
            x = torch.cat((x, x_adv))
            y = y.to(device) - min_y_train
            y = torch.cat((y, y))
            ans = model.forward(x)
            l = loss(ans, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            #             return ans, y
            return l.item()

        trainer = Engine(train_step)

        #         acc_metric.attach(trainer, "accuracy")
        #         loss_metric.attach(trainer, 'loss')

        def train_eval_step(engine, batch):
            model.eval()
            x, y = batch
            x = x.to(device)
            x_adv = torch.Tensor(attack.generate(x=x.cpu())).to(device)
            x = torch.cat((x, x_adv))
            y = y.to(device) - min_y_train
            y = torch.cat((y, y))
            with torch.no_grad():
                ans = model.forward(x)
            return ans, y

        train_evaluator = Engine(train_eval_step)
        acc_metric.attach(train_evaluator, "accuracy")
        loss_metric.attach(train_evaluator, 'loss')

        def validation_step(engine, batch):
            model.eval()
            x, y = batch
            x = x.to(device)
            x_adv = torch.Tensor(attack.generate(x=x.cpu())).to(device)
            x = torch.cat((x, x_adv))
            y = y.to(device) - min_y_val
            y = torch.cat((y, y))
            with torch.no_grad():
                ans = model.forward(x)
            return ans, y

        valid_evaluator = Engine(validation_step)
        acc_val_metric.attach(valid_evaluator, "accuracy")
        loss_val_metric.attach(valid_evaluator, 'loss')

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            valid_evaluator.run(ds_valid)
            metrics = valid_evaluator.state.metrics
            valid_avg_accuracy = metrics['accuracy']
            avg_nll = metrics['loss']
            print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                  .format(engine.state.epoch, valid_avg_accuracy, avg_nll))
            writer.add_scalar("validation/avg_loss",
                              avg_nll, engine.state.epoch)
            writer.add_scalar("validation/avg_accuracy",
                              valid_avg_accuracy, engine.state.epoch)
            writer.add_scalar("validation/avg_error", 1. -
                              valid_avg_accuracy, engine.state.epoch)

        @trainer.on(Events.EPOCH_COMPLETED)
        def lr_scheduler(engine):
            metrics = valid_evaluator.state.metrics
            avg_nll = metrics['accuracy']
            sched.step(avg_nll)

        @trainer.on(Events.ITERATION_COMPLETED(every=500))
        def log_training_loss(engine):
            batch = engine.state.batch
            ds = DataLoader(TensorDataset(*batch),
                            batch_size=batch_size)
            train_evaluator.run(ds)
            metrics = train_evaluator.state.metrics
            # metrics = engine.state.metrics
            accuracy = metrics['accuracy']
            nll = metrics['loss']
            iter = (engine.state.iteration - 1) % len(ds_train) + 1
            if (iter % 50) == 0:
                print("Epoch[{}] Iter[{}/{}] Accuracy: {:.2f} Loss: {:.2f}"
                      .format(engine.state.epoch, iter, len(ds_train), accuracy, nll))
            writer.add_scalar("batchtraining/detloss", nll, engine.state.epoch)
            writer.add_scalar("batchtraining/accuracy",
                              accuracy, engine.state.iteration)
            writer.add_scalar("batchtraining/error", 1. -
                              accuracy, engine.state.iteration)
            writer.add_scalar("batchtraining/loss",
                              engine.state.output, engine.state.iteration)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_lr(engine):
            writer.add_scalar(
                "lr", optimizer.param_groups[0]['lr'], engine.state.epoch)

#         @trainer.on(Events.EPOCH_COMPLETED)
#         def log_training_results(engine):
#             train_evaluator.run(ds_train)
#             metrics = train_evaluator.state.metrics
#             # metrics = engine.state.metrics
#             avg_accuracy = metrics['accuracy']
#             avg_nll = metrics['loss']
#             print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
#                   .format(engine.state.epoch, avg_accuracy, avg_nll))
#             writer.add_scalar("training/avg_loss", avg_nll, engine.state.epoch)
#             writer.add_scalar("training/avg_accuracy",
#                               avg_accuracy, engine.state.epoch)
#             writer.add_scalar("training/avg_error", 1. -
#                               avg_accuracy, engine.state.epoch)

        @trainer.on(Events.EPOCH_COMPLETED)
        def validation_value(engine):
            metrics = valid_evaluator.state.metrics
            valid_avg_accuracy = metrics['accuracy']
            return valid_avg_accuracy

        to_save = {'model': model}
        handler = Checkpoint(to_save, DiskSaver(os.path.join(base_data, model_id),
                                                create_dir=True),
                             score_function=validation_value, score_name="val_acc",
                             global_step_transform=global_step_from_engine(trainer),
                             n_saved=None)

        # kick everything off
        trainer.add_event_handler(Events.ITERATION_COMPLETED(every=200*5000//batch_size//5), handler)
        trainer.run(ds_train, max_epochs=max_epochs)

