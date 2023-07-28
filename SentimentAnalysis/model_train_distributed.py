import gc
import getopt
import math
import random
import sys
import time
import os

import numpy as np
import pandas as pd
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from CONSTANTS import *
from average_meter import AverageMeter
from bert_tweet_sparsemax import BertTweetWithSparsemax, RobertaSelfAttention
from bert_tweet_with_mask import BertTweetWithMask
from datasets import ReviewDataset, TwitterDataset


def parsing():
    # Remove 1st argument from the list of command line arguments
    arguments = sys.argv[1:]

    # Options
    options = "hc:m:b:e:d:f:"

    # Long options
    long_options = ["help", "cores=", "model=", "batch_size=", "epoch=", "dataset=", "filename="]

    # Prepare flags
    flags = {"cores": 1, "model": "robertaMask", "batch_size": TRAIN_BATCH_SIZE, "epoch": EPOCHS,
             "dataset": "twitter"}

    # Parsing argument
    arguments, values = getopt.getopt(arguments, options, long_options)

    if len(arguments) > 0 and arguments[0][0] in ("-h", "--help"):
        print(f'This script trains a model on a TPU with multiple cores.\n\
        -c or --cores: whether to train on a single core or on all available cores (default={flags["cores"]}).\n\
        -m or --model: model name, available options are {", ".join(MODEL_NAME_OPTIONS)} \
        (default={flags["model"]}).\n\
        -b or --batch_size: batch size used for training (default={TRAIN_BATCH_SIZE}).\n\
        -e or --epoch: number of epochs (default={EPOCHS}).\n\
        -d or --dataset: dataset name,  available options are {", ".join(DATASET_NAME_OPTIONS)}\
        (default={flags["dataset"]}).\n\
        -f or --filename: name of the file for the model (valid name with no extension).')
        sys.exit()

    # checking each argument
    for arg, val in arguments:
        if arg in ("-c", "--cores"):
            if int(val) <= 8:
                flags["cores"] = int(val)
            else:
                raise ValueError("Not enough xla devices.")
        if arg in ("-m", "--model"):
            if val in MODEL_NAME_OPTIONS:
                flags["model"] = val
            else:
                raise ValueError("Model argument not valid.")
        elif arg in ("-b", "--batch_size"):
            if int(val) >= 1:
                flags["batch_size"] = int(val)
            else:
                raise ValueError("Batch size must be at least 1.")
        elif arg in ("-e", "--epoch"):
            if int(val) >= 1:
                flags["epoch"] = int(val)
            else:
                raise ValueError("Number of epochs must be at least 1.")
        elif arg in ("-d", "--dataset"):
            if val in DATASET_NAME_OPTIONS:
                flags["dataset"] = val
            else:
                raise ValueError("Dataset name is not valid.")
        elif arg in ("-f", "--filename"):
            flags["filename"] = '{}.pt'.format(val)

    if "filename" not in flags.keys():
        flags["filename"] = '{}.pt'.format(flags["model"])

    return flags


def save_model_info(training_losses, eval_losses, training_accuracies, eval_accuracies, filename):
    # Make directory to store losses and accuracies if not present
    if not os.path.exists(PATH_LOSSES_AND_ACCURACIES):
        os.makedirs(PATH_LOSSES_AND_ACCURACIES)

    xm.master_print("- saving accuracies and losses...")
    xm.save(training_losses,  PATH_LOSSES_AND_ACCURACIES + "/" + 'trn-losses-{}.txt'.format(filename), master_only=True)
    xm.save(eval_losses,  PATH_LOSSES_AND_ACCURACIES + "/" + 'val-losses-{}.txt'.format(filename), master_only=True)
    xm.save(training_accuracies,  PATH_LOSSES_AND_ACCURACIES + "/" + 'trn-accuracies-{}.txt'.format(filename), master_only=True)
    xm.save(eval_accuracies,  PATH_LOSSES_AND_ACCURACIES + "/" + 'val-accuracies-{}.txt'.format(filename), master_only=True)
    xm.master_print("- saved!")


def get_training_and_validation_dataframes(path, dtype, grouping_key, train_fraction, eval_fraction, columns):
    # Load dataframe with dataset

    # Set random seed so the same dataset is sampled every time
    random_seed = 42
    random.seed(random_seed)

    df = pd.read_json(path, lines=True, dtype=dtype)
    dataset_size = df.shape[0]

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

    # Sample dataset for validation
    eval_values = []
    for key, group in grouped_df_after_drop.groups.items():
        for ind in random.sample(group.tolist(), k=math.ceil(eval_fraction * len(group))):
            eval_values.append((key, df_after_drop.iloc[ind, 1]))

    # Delete unused variable
    del df, grouped_df, df_after_drop, grouped_df_after_drop
    gc.collect()

    # Return training and validation dataframes plus the splits
    return pd.DataFrame(data=training_values, columns=columns), pd.DataFrame(data=eval_values, columns=columns), \
        len(training_values) / dataset_size, len(eval_values) / dataset_size


def _train_epoch_fn(model, para_loader, criterion, optimizer):
    # Set model for training
    model.train()

    # Define object for loss
    training_meter = AverageMeter()

    # Define list for accuracies and losses
    accuracies = []
    losses = []

    # Train for the epoch
    for i, data in enumerate(para_loader):
        # Set gradients to zero
        optimizer.zero_grad()
        # Get mini-batch data
        ids = data['input_ids']
        mask = data['attention_mask']
        cls_targets = data['cls_targets']
        # Compute model output
        outputs = model(ids, mask)
        # Compute accuracy

        # Compute loss
        loss = criterion(outputs, cls_targets)

        # Update running average of loss for epoch
        training_meter.update_loss(
            xm.mesh_reduce('loss_reduce', loss, lambda values: sum(values) / len(values)))

        # Update running average of accuracy for epoch
        training_meter.update_accuracy(
            xm.mesh_reduce(
                'accuracy_reduce',
                (torch.argmax(outputs, dim=1) == cls_targets).sum() / cls_targets.size(0),
                lambda values: sum(values) / len(values)
            )
        )

        # Feedback
        if i % VERBOSE_PARAM == 0:
            xm.master_print('-- step {} training | cur_loss = {:.6f}, avg_loss = {:.6f}, curr_acc = {:.6f}, avg_acc = '
                            '{:.6f}'
                            .format(i, training_meter.val_loss, training_meter.avg_loss, training_meter.val_accuracy,
                                    training_meter.avg_accuracy), flush=True)

        if i % VERBOSE_PARAM_FOR_SAVING == 0:
            # Save values
            accuracies.append((i, training_meter.val_accuracy))
            losses.append((i, training_meter.val_loss))

        # Compute gradient updates for learnable parameters
        loss.backward()
        # Update learnable parameters
        xm.optimizer_step(optimizer)

        # Free up memory
        del ids, mask, cls_targets, outputs, loss
        gc.collect()

    # Return accuracies and losses
    return accuracies, losses, training_meter.avg_accuracy, training_meter.avg_loss


def _eval_epoch_fn(model, para_loader, criterion):
    # Set model to evaluation
    model.eval()

    # Define object for loss
    eval_meter = AverageMeter()

    # Define list for accuracies and losses
    accuracies = []
    losses = []

    with torch.no_grad():
        for i, eval_batch in enumerate(para_loader):
            # Get mini-batch data
            eval_ids = eval_batch['input_ids']
            eval_mask = eval_batch['attention_mask']
            eval_cls_targets = eval_batch['cls_targets']
            # Compute model output
            eval_outputs = model(eval_ids, eval_mask)
            loss = criterion(eval_outputs, eval_cls_targets)

            # Update running average of loss for epoch
            eval_meter.update_loss(
                xm.mesh_reduce('loss_reduce', loss, lambda values: sum(values) / len(values)))

            # Update running average of accuracy for epoch
            eval_meter.update_accuracy(
                xm.mesh_reduce(
                    'accuracy_reduce',
                    (torch.argmax(eval_outputs, dim=1) == eval_cls_targets).sum() / eval_cls_targets.size(0),
                    lambda values: sum(values) / len(values)
                )
            )

            # Feedback
            if i % VERBOSE_PARAM == 0:
                # Print running average
                xm.master_print('-- step {} evaluation | cur_loss = {:.6f}, avg_loss = {:.6f}, curr_acc = {:.6f}, '
                                'avg_acc = {:.6f}'
                                .format(i, eval_meter.val_loss, eval_meter.avg_loss, eval_meter.val_accuracy,
                                        eval_meter.avg_accuracy), flush=True)

            # Save values every VERBOSE_PARAM steps
            if i % VERBOSE_PARAM_FOR_SAVING == 0:
                accuracies.append((i, eval_meter.val_accuracy))
                losses.append((i, eval_meter.val_loss))

            # Free up memory
            del eval_ids, eval_mask, eval_cls_targets, eval_outputs, loss
            gc.collect()

    # Return accuracies and losses
    return accuracies, losses, eval_meter.avg_accuracy, eval_meter.avg_loss


def _run(flags):
    # TOKENIZER

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    if flags["dataset"] == "amazon":
        tokenizer.add_tokens(AMAZON_TOKENS)

    # DATA FOR TRAINING AND EVALUATION

    if flags["dataset"] == "amazon":
        df_training, df_eval, training_frac, eval_frac = get_training_and_validation_dataframes(**AMAZON_OPTIONS)
        # Create train and eval datasets
        training_dataset = ReviewDataset(df_training, tokenizer)
        eval_dataset = ReviewDataset(df_eval, tokenizer)
    else:
        df_training, df_eval, training_frac, eval_frac = get_training_and_validation_dataframes(**TWITTER_OPTIONS)
        # Create train and eval datasets
        if flags["model"] == "robertaMask":
            training_dataset = TwitterDataset(df_training, tokenizer, use_embedder=True)
            eval_dataset = TwitterDataset(df_eval, tokenizer, use_embedder=True)
        else:
            training_dataset = TwitterDataset(df_training, tokenizer)
            eval_dataset = TwitterDataset(df_eval, tokenizer)

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
                                 batch_size=flags["batch_size"],
                                 sampler=train_sampler)

    eval_loader = DataLoader(eval_dataset,
                             batch_size=flags["batch_size"],
                             sampler=eval_sampler)

    # DEVICE

    device = xm.xla_device()

    # MODEL

    # Get model
    base_model = AutoModel.from_pretrained(MODEL)
    if flags["dataset"] == "amazon":
        base_model.resize_token_embeddings(len(tokenizer))

    if flags["model"] == "robertaMask":
        m = BertTweetWithMask(base_model)
        # Freeze parameters of all layers of the encoder except for the first and last layer
        for i in range(1, len(m.base_model.encoder.layer) - 1):
            for param in m.base_model.encoder.layer[i].parameters():
                param.requires_grad = False

    else:
        m = BertTweetWithSparsemax(base_model)
        # Change first and last self-attention layers of the model
        m.base_model.encoder.layer[0].attention.self = RobertaSelfAttention(config=m.base_model.config)
        #m.base_model.encoder.layer[1].attention.self = RobertaSelfAttention(config=m.base_model.config)
        #m.base_model.encoder.layer[-2].attention.self = RobertaSelfAttention(config=m.base_model.config)
        m.base_model.encoder.layer[-1].attention.self = RobertaSelfAttention(config=m.base_model.config)

        # Freeze parameters of all layers of the encoder except for the first and last layer
        #for param in m.base_model.embeddings.parameters():
        #    param.requires_grad = False
        #for i in range(2, len(m.base_model.encoder.layer) - 2):
        #    for param in m.base_model.encoder.layer[i].parameters():
        #        param.requires_grad = False

    model = m.to(device)

    # wait for all the tpus to reach here
    if not xm.is_master_ordinal():
        xm.rendezvous('load_model')

    if xm.is_master_ordinal():
        xm.rendezvous('load_model')

    xm.master_print('- model loaded!', flush=True)

    # LOSS FUNCTIONS AND OPTIMIZER

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    # LISTS FOR STORING LOSSES AND DICTIONARY FOR EARLY STOPPING

    training_losses = []
    eval_losses = []
    training_accuracies = []
    eval_accuracies = []
    early_stopping = {'best': np.Inf, 'no_improvement': 0, 'patience': PATIENCE, 'stop': False}

    # Print parameters before starting training
    xm.master_print(
        f'Training and evaluation of the model {flags["model"]} with early stopping. The parameters of the '
        + 'model are the following:+\n'
        + f'- Number of epochs: {flags["epoch"]}\n'
        + f'- Batch size: {flags["cores"] * flags["batch_size"]}\n'
        + f'- Training fraction: {training_frac:.6f}\n'
        + f'- Validation fraction: {eval_frac:.6f}\n'
        + f'- Dataset: {flags["dataset"]}\n'
        + f'- Patience: {early_stopping["patience"]}\n'
        + f'- Learning rate: {LEARNING_RATE}\n'
        + f'- Optimizer: Adam.', flush=True)

    # TRAINING

    for epoch in range(flags["epoch"]):
        if not early_stopping['stop']:
            # Update model parameter
            model.update_epoch(epoch + 1)
            # Display info
            xm.master_print('-' * 55, flush=True)
            xm.master_print('EPOCH {}/{}'.format(epoch + 1, EPOCHS), flush=True)
            xm.master_print('-' * 55, flush=True)
            xm.master_print('- initialization | TPU cores = {}'.format(xm.xrt_world_size()), flush=True)

            # Update training loader shuffling
            train_sampler.set_epoch(epoch)

            # Training pass
            train_start = time.time()
            xm.master_print('- training...', flush=True)
            para_loader = pl.ParallelLoader(training_loader, [device])
            new_accuracies, new_losses, training_accuracy, training_loss = _train_epoch_fn(model=model,
                                                                                           para_loader=para_loader
                                                                                           .per_device_loader(device),
                                                                                           criterion=criterion,
                                                                                           optimizer=optimizer)

            # Add new accuracies and losses
            training_accuracies += new_accuracies
            training_losses += new_losses

            del para_loader, new_accuracies, new_losses
            gc.collect()

            # Evaluation pass
            valid_start = time.time()
            xm.master_print('- validation...', flush=True)
            para_loader = pl.ParallelLoader(eval_loader, [device])
            new_accuracies, new_losses, eval_accuracy, eval_loss = _eval_epoch_fn(model=model,
                                                                                  para_loader=para_loader
                                                                                  .per_device_loader(device),
                                                                                  criterion=criterion)

            # Add new accuracies and losses
            eval_accuracies += new_accuracies
            eval_losses += new_losses

            del para_loader, new_accuracies, new_losses
            gc.collect()

            # Save weights
            if eval_loss < early_stopping['best']:
                xm.save(model.model_representation(), PATH_MODELS + "/" + flags["filename"], master_only=True)
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
            xm.master_print('- average accuracy | train = {:.6f}, valid = {:.6f}'.format(training_accuracy,
                                                                                         eval_accuracy),
                            flush=True)
            xm.master_print('-' * 55, flush=True)
            xm.master_print('', flush=True)

            del training_loss, eval_loss, training_accuracy, eval_accuracy
            gc.collect()

            # Save training and evaluation values collected up to the current epoch
            save_model_info(training_losses, eval_losses, training_accuracies, eval_accuracies, flags["filename"])

        else:
            xm.master_print('- early stopping triggered ', flush=True)
            break


def _map_fn(index, flags):
    """
    Function executed by all TPU cores.
    Args:
        index: Identifier of the worker node.
        flags: Parameters.

    Returns:

    """
    torch.set_default_tensor_type('torch.FloatTensor')
    _run(flags)


if __name__ == "__main__":
    # Define model and model wrapper
    flags = parsing()
    xmp.spawn(_map_fn, args=(flags,), nprocs=flags["cores"], start_method='fork')
