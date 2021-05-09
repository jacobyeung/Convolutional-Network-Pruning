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
from utils import *
from collections import Counter


def prune_train_loop(model, params, ds, dset, min_y, base_data, model_id, prune_type, device, batch_size, tpa, max_epochs=2):
    assert prune_type in ['global_unstructured', 'structured']
    total_prune_amount = tpa
    ds_train, ds_valid = ds
    train_set, valid_set = dset
    min_y_train, min_y_val = min_y
    model_id = f'{model_id}_{prune_type}_pruning_{tpa}'
    valid_freq = 200 * 500 // batch_size // 3
    
    conv_layers = [model.conv1]

    def prune_model(model):
#         remove_amount = total_prune_amount // (max_epochs)
        remove_amount = total_prune_amount
        print(f'pruned model by {remove_amount}')
        worst = select_filters(model, ds_valid, valid_set, remove_amount, device)
        worst = [k for k in Counter(torch.stack(worst).view(-1).cpu().numpy()).keys()]
        worst.sort(reverse=True)
        print(worst)
        for layer in conv_layers:
            for d in worst:
                TuckerStructured(layer, name='weight', amount=1, dim=0, filt=d)
        return worst
    bad = prune_model(model)
    zeros = []
    for i in range(len(model.conv1.weight_mask)):
        if torch.sum(model.conv1.weight_mask[i]) == 0.0:
            zeros.append(i)
    zeros.sort(reverse=True)
    if zeros == bad:
        print("correct;y zero'd filters")
    else:
        if len(zeros) == len(bad):
            for i in range(len(zeros)):
                if zeros[i] != bad[i]:
                    wrong.append((bad[i], zeros[i]))
        else:
            print("diff number filters zero'd", zeros)

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

        def train_step(engine, batch):
            model.train()
            x, y = batch
            x = x.to(device)
            y = y.to(device) - min_y_train
            optimizer.zero_grad()
            ans = model.forward(x)
            l = loss(ans, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            with torch.no_grad():
                for layer in conv_layers:
                    layer.weight *= layer.weight_mask  # make sure pruned weights stay 0
            return l.item()

        trainer = Engine(train_step)

        def train_eval_step(engine, batch):
            model.eval()
            x, y = batch
            x = x.to(device)
            y = y.to(device) - min_y_train
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
            with torch.no_grad():
                ans = model.forward(x)
            return ans, y

        valid_evaluator = Engine(validation_step)
        acc_val_metric.attach(valid_evaluator, "accuracy")
        loss_val_metric.attach(valid_evaluator, 'loss')

        @trainer.on(Events.ITERATION_COMPLETED(every=valid_freq))
#         @trainer.on(Events.ITERATION_COMPLETED)
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

#             prune_model(model)
            

        @trainer.on(Events.EPOCH_COMPLETED)
        def lr_scheduler(engine):
            metrics = valid_evaluator.state.metrics
            avg_nll = metrics['accuracy']
            sched.step(avg_nll)

        @trainer.on(Events.ITERATION_COMPLETED(every=50))
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