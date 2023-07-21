import gc
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from CONSTANTS import *
from average_meter import AverageMeter
from dataset import ReviewDataset
from model import Roberta


def plot_accuracy_and_loss(training_accuracies, training_losses, eval_accuracies, eval_losses):
    # Plot training and validation accuracy
    plt.plot(training_accuracies, label="Training Accuracy")
    plt.plot(eval_accuracies, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.savefig("accuracy.png")
    # Plot the training and validation loss
    plt.plot(training_losses, label="Training Loss")
    plt.plot(eval_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.savefig("loss.png")


def get_training_and_validation_dataframes(path, dtype, grouping_key, train_fraction, eval_fraction, columns):
    # Load dataframe with dataset
    df = pd.read_json(path, lines=True, dtype=dtype)

    # Group by overall
    grouped_df = df.groupby(grouping_key)

    # Sample 20 % of dataset for training
    training_values = []
    indeces_to_drop_training = []
    for key, group in grouped_df.groups.items():
        for ind in random.sample(group.tolist(), k=math.ceil(train_fraction * len(group))):
            training_values.append((key, df.iloc[ind, 1]))
            indeces_to_drop_training.append(ind)

    df_after_drop = df.drop(indeces_to_drop_training).reset_index(drop=True)
    grouped_df_after_drop = df_after_drop.groupby(grouping_key)

    # Sample 1 % of dataset for validation
    eval_values = []
    for key, group in grouped_df_after_drop.groups.items():
        for ind in random.sample(group.tolist(), k=math.ceil(eval_fraction * len(group))):
            eval_values.append((key, df_after_drop.iloc[ind, 1]))

    # Delete unused variable
    del df, grouped_df, df_after_drop, grouped_df_after_drop
    gc.collect()

    # Return training and validation dataframes
    return pd.DataFrame(data=training_values, columns=columns), pd.DataFrame(data=eval_values, columns=columns)


def _train_epoch_fn(epoch, model, para_loader, criterion, optimizer, device):
    # Set model for training
    model.train()

    # Define object for loss
    training_loss_meter = AverageMeter()

    # Train for the epoch
    for i, data in enumerate(para_loader):
        # Set gradients to zero
        optimizer.zero_grad()
        # Get mini-batch data
        ids = data['ids'].to(device)
        mask = data['mask'].to(device)
        cls_targets = data['cls_targets'].to(device)
        # Compute model output
        outputs = model(ids, mask)
        # Compute loss
        loss = criterion(outputs, cls_targets)
        # Update running average of loss for epoch
        training_loss_meter.update(xm.mesh_reduce('loss_reduce', loss, lambda values: sum(values) / len(values)),
                                   cls_targets.size(0))

        # feedback
        if i > 0 and i % VERBOSE_PARAM == 0:
            xm.master_print('-- step {} | cur_loss = {:.6f}, avg_loss = {:.6f}'.format(i, training_loss_meter.val,
                                                                                       training_loss_meter.avg),
                            flush=True)

        # Compute gradient updates for learnable parameters
        loss.backward()
        # Update learnable parameters
        xm.optimizer_step(optimizer)

        # Free up memory
        del ids, mask, cls_targets, outputs, loss
        gc.collect()

    # Return average training loss for the epoch
    return training_loss_meter.avg


def _eval_epoch_fn(epoch, model, para_loader, criterion, device):
    # Set model to evaluation
    model.eval()

    # Define object for loss
    eval_loss_meter = AverageMeter()

    with torch.no_grad():
        for eval_batch in para_loader:
            # Get mini-batch data
            eval_ids = eval_batch['ids'].to(device)
            eval_mask = eval_batch['mask'].to(device)
            eval_cls_targets = eval_batch['cls_targets'].to(device)
            # Compute model output
            eval_outputs = model(eval_ids, eval_mask)
            loss = criterion(eval_outputs, eval_cls_targets)

            eval_loss_meter.update(xm.mesh_reduce('loss_reduce', loss, lambda values: sum(values) / len(values)),
                                   eval_cls_targets.size(0))

            # Free up memory
            del eval_ids, eval_mask, eval_cls_targets, eval_outputs, loss
            gc.collect()

    # Output average evaluation loss for current epoch
    xm.master_print('-- epoch {} | avg_eval_loss =  {:.6f}'.format(epoch, eval_loss_meter.avg), flush=True)

    # Return average evaluation loss for the epoch
    return eval_loss_meter.avg


def _run():
    # TOKENIZER

    tokenizer = AutoTokenizer.from_pretrained(PATH_TOKENIZER)

    # DATA FOR TRAINING AND EVALUATION
    df_training, df_eval = get_training_and_validation_dataframes(PATH_DATASET,
                                                                  "overall int, reviewText string",
                                                                  "overall",
                                                                  DATASET_TRAIN_FRACTION,
                                                                  DATASET_EVAL_FRACTION,
                                                                  ["overall", "reviewText"])

    # Create train and eval datasets
    training_dataset = ReviewDataset(df_training, tokenizer)
    eval_dataset = ReviewDataset(df_eval, tokenizer)

    # Create data samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(training_dataset,
                                                                    num_replicas=xm.xrt_world_size(),
                                                                    rank=xm.get_ordinal(),
                                                                    shuffle=True)

    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset,
                                                                   num_replicas=xm.xrt_world_size(),
                                                                   rank=xm.get_ordinal(),
                                                                   shuffle=False)

    # Create dataloaders
    training_loader = DataLoader(training_dataset,
                                 batch_size=TRAIN_BATCH_SIZE,
                                 shuffle=True,
                                 sampler=train_sampler,
                                 num_workers=0)

    eval_loader = DataLoader(eval_dataset,
                             batch_size=TRAIN_BATCH_SIZE,
                             shuffle=False,
                             sampler=eval_sampler,
                             num_workers=0)
    # DEVICE

    device = xm.xla_device()

    # MODEL

    model = Roberta()
    model.to(device)

    # LOSS FUNCTIONS AND OPTIMIZER

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    # LISTS FOR STORING LOSSES AND DICTIONARY FOR EARLY STOPPING

    training_losses = []
    eval_losses = []
    early_stopping = {'best': np.Inf, 'no_improvement': 0, 'patience': 2, 'stop': False}

    # TRAINING

    for epoch in range(EPOCHS):
        if not early_stopping['stop']:
            # Display info
            xm.master_print('-' * 55, flush=True)
            xm.master_print('EPOCH {}/{}'.format(epoch + 1, EPOCHS), flush=True)
            xm.master_print('-' * 55, flush=True)
            xm.master_print('- initialization | TPU cores = {}'.format(xm.xrt_world_size()), flush=True)
            epoch_start = time.time()

            # Update training loader shuffling
            train_sampler.set_epoch(epoch)

            # Training pass
            train_start = time.time()
            xm.master_print('- training...')
            para_loader = pl.ParallelLoader(training_loader, [device])
            training_loss = _train_epoch_fn(epoch=epoch + 1,
                                            model=model,
                                            para_loader=para_loader.per_device_loader(device),
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            device=device)

            del para_loader
            gc.collect()

            # Evaluation pass
            valid_start = time.time()
            xm.master_print('- validation...', flush=True)
            para_loader = pl.ParallelLoader(eval_loader, [device])
            eval_loss = _eval_epoch_fn(epoch=epoch + 1,
                                       model=model,
                                       para_loader=para_loader.per_device_loader(device),
                                       criterion=criterion,
                                       device=device)

            del para_loader
            gc.collect()

            # Save weights
            if eval_loss < early_stopping['best']:
                xm.save(model.state_dict(), 'weights_{}.pt'.format(MODEL_NAME))
                early_stopping['best'] = eval_loss
                early_stopping['no_improvement'] = 0
            else:
                early_stopping['no_improvement'] += 1
                if early_stopping['no_improvement'] == early_stopping['patience']:
                    early_stopping['stop'] = True

            # Display info
            xm.master_print('- elapsed time | train = {:.2f} min, valid = {:.2f} min'.format(
                (valid_start - train_start) / 60, (time.time() - valid_start) / 60), flush=True)
            xm.master_print('- average loss | train = {:.6f}, valid = {:.6f}'.format(training_loss, eval_loss),
                            flush=True)
            xm.master_print('-' * 55)
            xm.master_print('')

            # save losses
            training_losses.append(training_loss)
            eval_losses.append(eval_loss)

            del training_loss, eval_loss
            gc.collect()

        else:
            xm.master_print('- early stopping triggered ', flush=True)
            break

    return training_losses, eval_losses


def _map_fn(index, flags):
    """
    Function executed by all TPU cores.
    Args:
        index: Identifier of the worker node.
        flags: Parameters.

    Returns:

    """
    training_losses, eval_losses = _run()
    np.save('trn_losses.npy', np.array(training_losses))
    np.save('val_losses.npy', np.array(eval_losses))


if __name__ == "__main__":
    # Define model and model wrapper
    flags = {}
    xmp.spawn(_map_fn, args=(flags,), nprocs=NUMBER_OF_TPU_WORKERS, start_method='fork')
