import os
import pickle
from utils import create_summary_writer
import torch
import ignite
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import (Engine, Events, create_supervised_trainer,
                           create_supervised_evaluator)
from ignite.handlers import ModelCheckpoint, EarlyStopping, Checkpoint, DiskSaver
from ignite.metrics import Accuracy, Loss
import torch.nn.functional as F


def train_loop(model, params, ds, base_data, model_id, device="cpu", max_epochs=100):
    ds_train, ds_valid = ds

    with create_summary_writer(model, ds_train, base_data, model_id) as writer:
        lr = params['lr']
        mom = params['momentum']
        wd = params['l2_wd']
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=mom, weight_decay=wd)
        sched = ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
        funcs = {'accuracy': Accuracy(),
                 'loss': Loss(F.cross_entropy)}
        loss = funcs['loss']._loss_fn
        trainer = create_supervised_trainer(model, optimizer, loss,
                                            device=device)
        train_evaluator = create_supervised_evaluator(
            model, metrics=funcs, device=device)
        valid_evaluator = create_supervised_evaluator(
            model, metrics=funcs, device=device)
        test_evaluator = create_supervised_evaluator(
            model, metrics=funcs, device=device)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            valid_evaluator.run(ds_valid)
            metrics = valid_evaluator.state.metrics
            valid_avg_accuracy = metrics['accuracy']
            avg_nll = metrics['loss']
            print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                  .format(engine.state.epoch, valid_avg_accuracy, avg_nll))
            writer.add_scalar("valdation/avg_loss",
                              avg_nll, engine.state.epoch)
            writer.add_scalar("valdation/avg_accuracy",
                              valid_avg_accuracy, engine.state.epoch)
            writer.add_scalar("valdation/avg_error", 1. -
                              valid_avg_accuracy, engine.state.epoch)

        @trainer.on(Events.EPOCH_COMPLETED)
        def lr_scheduler(engine):
            metrics = valid_evaluator.state.metrics
            avg_nll = metrics['accuracy']
            sched.step(avg_nll)

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            batch = engine.state.batch
            ds = DataLoader(TensorDataset(*batch),
                            batch_size=params['batch_size'])
            train_evaluator.run(ds)
            metrics = train_evaluator.state.metrics
            accuracy = metrics['accuracy']
            nll = metrics['loss']
            iter = (engine.state.iteration - 1) % len(ds_train) + 1
            if (iter % 100) == 0:
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

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            train_evaluator.run(ds_train)
            metrics = train_evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_nll = metrics['loss']
            print("Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                  .format(engine.state.epoch, avg_accuracy, avg_nll))
            writer.add_scalar("training/avg_loss", avg_nll, engine.state.epoch)
            writer.add_scalar("training/avg_accuracy",
                              avg_accuracy, engine.state.epoch)
            writer.add_scalar("training/avg_error", 1. -
                              avg_accuracy, engine.state.epoch)

        @trainer.on(Events.EPOCH_COMPLETED)
        def validation_value(engine):
            metrics = valid_evaluator.state.metrics
            valid_avg_accuracy = metrics['accuracy']
            return valid_avg_accuracy

        checkpoint = ModelCheckpoint(os.path.join(base_data, model_id), model_id,
                                     score_function=validation_value,
                                     score_name='valid_{}'.format('accuracy'))
        early_stopping = EarlyStopping(20, score_function=validation_value,
                                       trainer=trainer)
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, checkpoint, {'model': model})
        valid_evaluator.add_event_handler(Events.COMPLETED, early_stopping)

        to_save = {'trainer': trainer, 'model': model,
                   'optimizer': optimizer, 'lr_scheduler': sched}
        handler = Checkpoint(to_save, DiskSaver(os.path.join(
            base_data, "resume_training"), create_dir=True))
        # kick everything off
        trainer.add_event_handler(Events.EPOCH_COMPLETED, handler)
        trainer.run(ds_train, max_epochs=max_epochs)
