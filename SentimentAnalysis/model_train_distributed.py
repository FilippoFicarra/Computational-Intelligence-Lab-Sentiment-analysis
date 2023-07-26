import gc
import getopt
import sys
import time

import numpy as np
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
from dataset import TwitterDataset


def parsing():
    # Remove 1st argument from the list of command line arguments
    arguments = sys.argv[1:]

    # Options
    options = "hc:m:b:e:d:n:"

    # Long options
    long_options = ["help", "cores=", "model=", "batch_size=", "epoch=", "dataset=", "filename="]

    # Prepare flags
    flags = {"cores": 1, "model": "robertaMask", "batch_size": TRAIN_BATCH_SIZE, "epoch": EPOCHS,
             "dataset": "twitter"}

    # Parsing argument
    arguments, values = getopt.getopt(arguments, options, long_options)

    if len(arguments) > 0 and arguments[0][0] in ("-h", "--help"):
        print(f"""This script trains a model on a TPU with multiple cores.\n
        -c or --cores: whether to train on a single core or on all available cores (default={flags["cores"]}).\n
        -m or --model: model name, available options are {", ".join(MODEL_NAME_OPTIONS)} 
        (default={flags["model"]}).\n
        -b or --batch_size: batch size used for training (default={TRAIN_BATCH_SIZE}).\n
        -e or --epoch: number of epochs (default={EPOCHS}).\n
        -d or --dataset: dataset name,  available options is "twitter (default={flags["dataset"]}).\n
        -n or --filename: name of the file for the model (valid name with no extension).
        """)
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
            if val is not "twitter":
                flags["dataset"] = val
            else:
                raise ValueError("Dataset name is not valid.")
        elif arg in ("-n", "--filename"):
            flags["filename"] = '{}.pt'.format(val)

        if "filename" not in flags.keys():
            flags["filename"] = '{}.pt'.format(flags["model"])

    return flags


def save_model_info(training_losses, eval_losses, training_accuracies, eval_accuracies, filename):
    xm.master_print("- saving accuracies and losses...")
    xm.save(training_losses, 'trn-losses-{}.txt'.format(filename), master_only=True)
    xm.save(eval_losses, 'val-losses-{}.txt'.format(filename), master_only=True)
    xm.save(training_accuracies, 'trn-accuracies-{}.txt'.format(filename), master_only=True)
    xm.save(eval_accuracies, 'val-accuracies-{}.txt'.format(filename), master_only=True)
    xm.master_print("- saved!")


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
        training_meter.update_loss(xm.mesh_reduce('loss_reduce', loss.item(), lambda values: sum(values) / len(values)))

        # Update running average of accuracy for epoch
        training_meter.update_accuracy(
            xm.mesh_reduce(
                'accuracy_reduce',
                (torch.argmax(outputs, dim=1) == cls_targets).sum().item() / cls_targets.size(0),
                lambda values: sum(values) / len(values)
            )
        )

        # Feedback
        if i % VERBOSE_PARAM == 0:
            xm.master_print('-- step {} | cur_loss = {:.6f}, avg_loss = {:.6f}, curr_acc = {:.6f}, avg_acc = {:.6f}'
                            .format(i, training_meter.val_loss, training_meter.avg_loss, training_meter.val_accuracy,
                                    training_meter.avg_accuracy), flush=True)
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
            eval_meter.update_loss(xm.mesh_reduce('loss_reduce', loss.item(), lambda values: sum(values) / len(values)))

            # Update running average of accuracy for epoch
            eval_meter.update_accuracy(
                xm.mesh_reduce(
                    'accuracy_reduce',
                    (torch.argmax(eval_outputs, dim=1) == eval_cls_targets).sum().item() / eval_cls_targets.size(0),
                    lambda values: sum(values) / len(values)
                )
            )

            # Save values every VERBOSE_PARAM steps
            if i % VERBOSE_PARAM == 0:
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

    # DEVICE

    device = xm.xla_device()

    if flags["dataset"] == "amazon":
        tokenizer.add_tokens(SPECIAL_TOKENS_AMAZON)

    # DATA FOR TRAINING AND EVALUATION

    xm.master_print('- twitter dataset.', flush=True)
    training = torch.load(TENSOR_TRAINING_DATA_PATH)
    evaluation = torch.load(TENSOR_EVAL_DATA_PATH)

    # Create train and eval datasets
    if flags["model"] == "robertaMask":
        training_dataset = TwitterDataset(training, device)
        eval_dataset = TwitterDataset(evaluation, device)
    else:
        training_dataset = TwitterDataset(training, device)
        eval_dataset = TwitterDataset(evaluation, device)

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

    train_device_loader = pl.MpDeviceLoader(training_loader, device)

    test_device_loader = pl.MpDeviceLoader(eval_loader, device)

    # MODEL

    # Get model
    base_model = AutoModel.from_pretrained(MODEL)
    if flags["dataset"] == "amazon":
        base_model.resize_token_embeddings(len(tokenizer))

    if flags["model"] == "robertaMask":
        m = BertTweetWithMask(base_model)
        # Freeze parameters of all layers of the encoder except for the first and last layer
        # for param in m.base_model.embeddings.parameters():
        #     param.requires_grad = False
        # for i in range(2, len(m.base_model.encoder.layer) - 2):
        #     for param in m.base_model.encoder.layer[i].parameters():
        #         param.requires_grad = False

    else:
        m = BertTweetWithSparsemax(base_model)
        # Change first and last self-attention layers of the model
        m.base_model.encoder.layer[0].attention.self = RobertaSelfAttention(config=m.base_model.config)
        m.base_model.encoder.layer[1].attention.self = RobertaSelfAttention(config=m.base_model.config)
        m.base_model.encoder.layer[-2].attention.self = RobertaSelfAttention(config=m.base_model.config)
        m.base_model.encoder.layer[-1].attention.self = RobertaSelfAttention(config=m.base_model.config)

        # Freeze parameters of all layers of the encoder except for the first and last layer
        for param in m.base_model.embeddings.parameters():
            param.requires_grad = False
        for i in range(2, len(m.base_model.encoder.layer) - 2):
            for param in m.base_model.encoder.layer[i].parameters():
                param.requires_grad = False

    model = m.to(device)

    # LOSS FUNCTIONS AND OPTIMIZER

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    # LISTS FOR STORING LOSSES AND DICTIONARY FOR EARLY STOPPING

    training_losses = []
    eval_losses = []
    training_accuracies = []
    eval_accuracies = []
    early_stopping = {'best': np.Inf, 'no_improvement': 0, 'patience': PATIENCE, 'stop': False}

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
            xm.master_print('- training...')
            new_accuracies, new_losses, training_accuracy, training_loss = _train_epoch_fn(model=model,
                                                                                           para_loader=train_device_loader,
                                                                                           criterion=criterion,
                                                                                           optimizer=optimizer)

            # Add new accuracies and losses
            training_accuracies += new_accuracies
            training_losses += new_losses

            del new_accuracies, new_losses
            gc.collect()

            # Evaluation pass
            valid_start = time.time()
            xm.master_print('- validation...', flush=True)
            new_accuracies, new_losses, eval_accuracy, eval_loss = _eval_epoch_fn(model=model,
                                                                                  para_loader=test_device_loader,
                                                                                  criterion=criterion)

            # Add new accuracies and losses
            eval_accuracies += new_accuracies
            eval_losses += new_losses

            del new_accuracies, new_losses
            gc.collect()

            # Save weights
            if eval_loss < early_stopping['best']:
                xm.save(model.model_representation(), "model/" + flags["filename"], master_only=True)
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
            xm.master_print('-' * 55)
            xm.master_print('')

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
