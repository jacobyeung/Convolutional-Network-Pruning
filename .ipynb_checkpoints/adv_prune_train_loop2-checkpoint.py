import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import ignite
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.metrics import Accuracy, Loss
from advertorch.context import ctx_noparamgrad_and_eval
from advertorch.attacks import GradientSignAttack
from utils import *
from tqdm import tqdm

def adv_prune_train_loop(model, params, ds, dset, min_y, base_data, model_id, prune_type, device, batch_size, tpa, max_epochs=1):
#     assert prune_type in ['global_unstructured', 'structured']
    ds_train, ds_valid = ds
    min_y_train, min_y_val = min_y
    train_set, valid_set = dset
    total_prune_amount = tpa
    original_model = copy.deepcopy(model)
    original_model.eval()
    model_id = f'{model_id}_{prune_type}_pruning_{tpa}_l1'
    valid_freq = 200 * 500 // batch_size // 3
    conv_layers = [model.conv1]
    for sequential in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for bottleneck in sequential:
            conv_layers.extend([bottleneck.conv1, bottleneck.conv2, bottleneck.conv3])
    conv_layers = conv_layers[:22]
    def prune_model(model):
        print(f'pruned model by {total_prune_amount}')
        if prune_type == 'global_unstructured':
            parameters_to_prune = [(layer, 'weight') for layer in conv_layers]
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=total_prune_amount,
            )
        else:
            for layer in conv_layers:
                prune.ln_structured(layer, name='weight', amount=total_prune_amount, n=1, dim=0)

    prune_model(model)

    def valid_eval(model, dataset, dataloader, device, label):
        right = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader), total=len(dataset) / dataloader.batch_size):
                data, y = data
                data = data.to(device)
                y = y.to(device) - label
                ans = model.forward(data)
                right += torch.sum(torch.eq(torch.argmax(ans, dim=1), y))
                total += y.shape[0]
        return right/total
    valid_acc = valid_eval(model, valid_set, ds_valid, device, min_y_val)
    print('initial accuracy:', valid_acc.item())
    
    with create_summary_writer(model, ds_train, base_data, model_id, device=device) as writer:
        lr = params['lr']
        mom = params['momentum']
        wd = params['l2_wd']
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
        sched = ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        funcs = {'accuracy': Accuracy(), 'loss': Loss(F.cross_entropy)}
        loss = funcs['loss']._loss_fn

        acc_metric = Accuracy(device=device)
        loss_metric = Loss(F.cross_entropy, device=device)

        acc_val_metric = Accuracy(device=device)
        loss_val_metric = Loss(F.cross_entropy, device=device)

#         attack = GradientSignAttack(original_model, loss_fn=loss, eps=0.2)

        def train_step(engine, batch):
            model.train()
            x, y = batch
            x = x.to(device)
            y = y.to(device) - min_y_train
#             with ctx_noparamgrad_and_eval(model):
#                 x_adv = attack.perturb(x, y)
#             optimizer.zero_grad()
#             x = torch.cat((x, x_adv))
#             y = torch.cat((y, y))
            ans = model.forward(x)
            l = loss(ans, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            with torch.no_grad():
                for layer in conv_layers:
                    layer.weight *= layer.weight_mask
            return l.item()

        trainer = Engine(train_step)

        def train_eval_step(engine, batch):
            model.eval()
            x, y = batch
            x = x.to(device)
            y = y.to(device) - min_y_train
#             x_adv = attack.perturb(x, y)
#             x = torch.cat((x, x_adv))
#             y = torch.cat((y, y))
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
            y = y.to(device) - min_y_val
#             x_adv = attack.perturb(x, y)
#             x = torch.cat((x, x_adv))
#             y = torch.cat((y, y))
            with torch.no_grad():
                ans = model.forward(x)
            return ans, y

        valid_evaluator = Engine(validation_step)
        acc_val_metric.attach(valid_evaluator, "accuracy")
        loss_val_metric.attach(valid_evaluator, 'loss')

        @trainer.on(Events.ITERATION_COMPLETED(every=valid_freq))
        def log_validation_results(engine):
            valid_evaluator.run(ds_valid)
            metrics = valid_evaluator.state.metrics
            valid_avg_accuracy = metrics['accuracy']
            avg_nll = metrics['loss']
            print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                  .format(engine.state.epoch, valid_avg_accuracy, avg_nll))
            writer.add_scalar("validation/avg_loss", avg_nll, engine.state.epoch)
            writer.add_scalar("validation/avg_accuracy", valid_avg_accuracy, engine.state.epoch)
            writer.add_scalar("validation/avg_error", 1. - valid_avg_accuracy, engine.state.epoch)

        @trainer.on(Events.EPOCH_COMPLETED)
        def lr_scheduler(engine):
            metrics = valid_evaluator.state.metrics
            avg_nll = metrics['accuracy']
            sched.step(avg_nll)

        @trainer.on(Events.ITERATION_COMPLETED(every=100))
        def log_training_loss(engine):
            batch = engine.state.batch
            ds = DataLoader(TensorDataset(*batch), batch_size=batch_size)
            train_evaluator.run(ds)
            metrics = train_evaluator.state.metrics
            accuracy = metrics['accuracy']
            nll = metrics['loss']
            iter = (engine.state.iteration - 1) % len(ds_train) + 1
            if (iter % 50) == 0:
                print("Epoch[{}] Iter[{}/{}] Accuracy: {:.2f} Loss: {:.2f}"
                      .format(engine.state.epoch, iter, len(ds_train), accuracy, nll))
            writer.add_scalar("batchtraining/detloss", nll, engine.state.epoch)
            writer.add_scalar("batchtraining/accuracy", accuracy, engine.state.iteration)
            writer.add_scalar("batchtraining/error", 1. - accuracy, engine.state.iteration)
            writer.add_scalar("batchtraining/loss", engine.state.output, engine.state.iteration)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_lr(engine):
            writer.add_scalar("lr", optimizer.param_groups[0]['lr'], engine.state.epoch)

        @trainer.on(Events.ITERATION_COMPLETED(every=valid_freq))
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
        trainer.add_event_handler(Events.ITERATION_COMPLETED(every=valid_freq), handler)
        trainer.run(ds_train, max_epochs=max_epochs)
