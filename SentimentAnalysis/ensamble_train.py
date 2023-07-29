"""This module trains the weights of an ensamble of models. """

import getopt
import sys
import time

import numpy as np
import torch
# import torch_xla.core.xla_model as xm
from torch.nn import Module, Parameter
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PretrainedConfig
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings, RobertaLayer

from average_meter import AverageMeter
from datasets import TwitterDatasetEnsamble
from utility_functions import *


class LinearCombinationModel(Module):
    def __init__(self):
        super().__init__()
        self.ensamble_models = []

        # Define weights for each model
        self.weight1 = Parameter(torch.tensor(1.0))
        self.weight2 = Parameter(torch.tensor(1.0))
        self.weight3 = Parameter(torch.tensor(1.0))

        self.epoch = 0

    def forward(self, input1, input2, input3):
        # Calculate the linear combinations
        linear_combination = self.weight1 * input1 + self.weight2 * input2 + self.weight3 * input3

        return linear_combination

    def update_epoch(self, epoch):
        self.epoch = epoch

    def model_representation(self):
        return {
            'epoch': self.epoch,
            'model_state_dict': self.state_dict()
        }

    def load_model(self, file_path):
        state = torch.load(file_path)
        self.epoch = state['epoch']
        self.load_state_dict(state['model_state_dict'])


class EnsamblerWithSelfAttention(Module):
    def __init__(self):
        super().__init__()
        config = PretrainedConfig.from_pretrained(MODEL)
        self.embeddings = RobertaEmbeddings(config)
        self.self_attention = RobertaLayer(config)
        self.linear = torch.nn.Linear(HIDDEN_SIZE, 3)

        self.epoch = 0

    def forward(self, input_ids, input1, input2, input3):
        # Calculate the linear combinations
        embeddings = self.embeddings(input_ids=input_ids)
        self_attention_out = self.self_attention(embeddings)
        weights = self.linear(self_attention_out[0][:, 0, :])

        return weights[:, 0].unsqueeze(1) * input1 + weights[:, 1].unsqueeze(1) * input2 + weights[:, 2].unsqueeze(1) * input3

    def update_epoch(self, epoch):
        self.epoch = epoch

    def model_representation(self):
        return {
            'epoch': self.epoch,
            'model_state_dict': self.state_dict()
        }

    def load_model(self, file_path):
        state = torch.load(file_path)
        self.epoch = state['epoch']
        self.load_state_dict(state['model_state_dict'])


def parsing():
    # Remove 1st argument from the list of command line arguments
    arguments = sys.argv[1:]

    # Options
    options = "hm:b:e:"

    # Long options
    long_options = ["help", "model=", "batch_size=", "epoch="]

    # Prepare flags
    flags = {"model": "self_attention", "batch_size": TRAIN_BATCH_SIZE, "epoch": EPOCHS}

    # Parsing argument
    arguments, values = getopt.getopt(arguments, options, long_options)

    if len(arguments) > 0 and arguments[0][0] in ("-h", "--help"):
        print(f'This script trains an ensamble on a TPU.\n\
        -m or --model: model name, available options are {", ".join(MODEL_NAME_OPTIONS_ENSAMBLE)} '
              f'(default={flags["model"]}).\n\
        -b or --batch_size: batch size used for training (default={TRAIN_BATCH_SIZE}).\n\
        -e or --epoch: number of epochs (default={EPOCHS}).')
        sys.exit()

    # checking each argument
    for arg, val in arguments:
        if arg in ("-m", "--model"):
            if val in MODEL_NAME_OPTIONS_ENSAMBLE:
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

    flags["filename"] = 'model-ensamble-{}.pt'.format(random.randint(1, 100000))

    return flags


def save_model_info(training_losses, eval_losses, training_accuracies, eval_accuracies, filename):
    # Make directory to store losses and accuracies if not present
    if not os.path.exists(PATH_LOSSES_AND_ACCURACIES):
        os.makedirs(PATH_LOSSES_AND_ACCURACIES)

    print("- saving accuracies and losses...")
    torch.save(training_losses, PATH_LOSSES_AND_ACCURACIES + "/" + 'trn-losses-{}.txt'.format(filename))
    torch.save(eval_losses, PATH_LOSSES_AND_ACCURACIES + "/" + 'val-losses-{}.txt'.format(filename))
    torch.save(training_accuracies, PATH_LOSSES_AND_ACCURACIES + "/" + 'trn-accuracies-{}.txt'.format(filename))
    torch.save(eval_accuracies, PATH_LOSSES_AND_ACCURACIES + "/" + 'val-accuracies-{}.txt'.format(filename))
    print("- saved!")


def _train_epoch_fn(model, ensamble, loader, criterion, optimizer, device, flags):
    # Set model for training
    model.train()

    # Define object for loss
    training_meter = AverageMeter()

    # Define list for accuracies and losses
    accuracies = []
    losses = []

    # Train for the epoch
    for i, data in enumerate(loader):
        # Set gradients to zero
        optimizer.zero_grad()
        # Get mini-batch data
        ids = data['input_ids'].to(device)
        mask = data['attention_mask'].to(device)
        ids_masker = data['input_ids_masker'].to(device)
        mask_masker = data['attention_mask_masker'].to(device)
        cls_targets = data['cls_targets'].to(device)

        # Compute outputs of the models in the ensamble
        outputs = []
        for e_model, requires_mask in ensamble:
            if requires_mask:
                outputs.append(e_model(ids_masker, mask_masker))
            else:
                outputs.append(e_model(ids, mask))

        # Compute model output
        if flags["model"] == "linear":
            output_ensamble = model(outputs[0], outputs[1], outputs[2])
        else:
            output_ensamble = model(ids, outputs[0], outputs[1], outputs[2])

        # Compute loss
        loss = criterion(output_ensamble, cls_targets)

        # Update running average of loss for epoch
        training_meter.update_loss(loss.item())

        # Update running average of accuracy for epoch
        training_meter.update_accuracy((torch.argmax(output_ensamble, dim=1) == cls_targets)
                                       .sum().item() / cls_targets.size(0))

        # Feedback
        if i % VERBOSE_PARAM == 0:
            print('-- step {} training | cur_loss = {:.6f}, avg_loss = {:.6f}, curr_acc = {:.6f}, avg_acc = '
                  '{:.6f}'.format(i, training_meter.val_loss, training_meter.avg_loss,
                                  training_meter.val_accuracy, training_meter.avg_accuracy))

        if i % VERBOSE_PARAM_FOR_SAVING == 0:
            # Save values
            accuracies.append((i, training_meter.val_accuracy))
            losses.append((i, training_meter.val_loss))

        # Compute gradient updates for learnable parameters
        loss.backward()
        # Update learnable parameters
        optimizer.step()
        # If the device being used is the tpu, take step
        # xm.mark_step()

        # Free up memory
        del ids, mask, cls_targets, outputs, loss
        gc.collect()

    # Return accuracies and losses
    return accuracies, losses, training_meter.avg_accuracy, training_meter.avg_loss


def _eval_epoch_fn(model, ensamble, loader, criterion, device, flags):
    # Set model to evaluation
    model.eval()

    # Define object for loss
    eval_meter = AverageMeter()

    # Define list for accuracies and losses
    accuracies = []
    losses = []

    with torch.no_grad():
        for i, eval_batch in enumerate(loader):
            # Get mini-batch data
            eval_ids = eval_batch['input_ids'].to(device)
            eval_mask = eval_batch['attention_mask'].to(device)
            eval_ids_masker = eval_batch['input_ids_masker'].to(device)
            eval_mask_masker = eval_batch['attention_mask_masker'].to(device)
            eval_cls_targets = eval_batch['cls_targets'].to(device)

            # Compute outputs of the models in the ensamble
            outputs = []
            for e_model, requires_mask in ensamble:
                if requires_mask:
                    outputs.append(e_model(eval_ids_masker, eval_mask_masker))
                else:
                    outputs.append(e_model(eval_ids, eval_mask))

            # Compute model output
            if flags["model"] == "linear":
                eval_outputs = model(outputs[0], outputs[1], outputs[2])
            else:
                eval_outputs = model(eval_ids, outputs[0], outputs[1], outputs[2])

            # Compute loss
            loss = criterion(eval_outputs, eval_cls_targets)

            # Update running average of loss for epoch
            eval_meter.update_loss(loss.item())

            # Update running average of accuracy for epoch
            eval_meter.update_accuracy((torch.argmax(eval_outputs, dim=1) == eval_cls_targets)
                                       .sum().item() / eval_cls_targets.size(0))

            # Feedback
            if i % VERBOSE_PARAM == 0:
                # Print running average
                print('-- step {} evaluation | cur_loss = {:.6f}, avg_loss = {:.6f}, curr_acc = {:.6f}, '
                      'avg_acc = {:.6f}'.format(i, eval_meter.val_loss, eval_meter.avg_loss,
                                                eval_meter.val_accuracy, eval_meter.avg_accuracy))

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
    # Get train and validation dataframes
    df_training, df_eval, training_frac, eval_frac = get_training_and_validation_dataframes(**TWITTER_OPTIONS)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Create datasets and data loaders
    training_dataset = TwitterDatasetEnsamble(df_training, tokenizer)
    eval_dataset = TwitterDatasetEnsamble(df_eval, tokenizer)

    training_loader = DataLoader(training_dataset,
                                 batch_size=flags["batch_size"],
                                 shuffle=True)

    eval_loader = DataLoader(eval_dataset,
                             batch_size=flags["batch_size"],
                             shuffle=False)

    # DEVICE

    # device = xm.xla_device()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MODEL

    # Upload all ensamble_models for making predictions. Models are freezed.
    ensamble_models = []
    # Check if the folder exists
    if not os.path.exists(PATH_MODELS):
        raise Exception("Folder does not exist.")
    else:
        # Check if the folder is empty
        files_in_folder = os.listdir(PATH_MODELS)
        if not files_in_folder:
            raise Exception("Folder is empty.")
        else:
            # Filter the files to get only the ones ending with ".pt"
            pt_files = [file for file in files_in_folder if file.endswith(".pt")]

            if len(pt_files) != 3:
                raise Exception("This module only works with ensamble of three models.")

            else:
                # Loop over the .pt files in the folder
                for pt_file in sorted(pt_files):
                    # Get model
                    trained_model, mask = get_model(pt_file, device)
                    # Set model for evaluation
                    trained_model.eval()
                    ensamble_models.append((trained_model, mask))

    # Create model for ensamble
    if flags["model"] == "linear":
        model = LinearCombinationModel()
    else:
        model = EnsamblerWithSelfAttention()

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    # LISTS FOR STORING LOSSES AND DICTIONARY FOR EARLY STOPPING

    training_losses = []
    eval_losses = []
    training_accuracies = []
    eval_accuracies = []
    early_stopping = {'best': np.Inf, 'no_improvement': 0, 'patience': PATIENCE, 'stop': False}

    # Print parameters before starting training
    print(
        f'Training and evaluation of the model ensamble with early stopping. The parameters of the '
        + 'model are the following:+\n'
        + f'- Number of epochs: {flags["epoch"]}\n'
        + f'- Batch size: {flags["batch_size"]}\n'
        + f'- Training fraction: {training_frac:.6f}\n'
        + f'- Validation fraction: {eval_frac:.6f}\n'
        + f'- Patience: {early_stopping["patience"]}\n'
        + f'- Learning rate: {LEARNING_RATE}\n'
        + f'- Optimizer: Adam.', flush=True)

    # TRAINING

    # EPOCHS LOOP

    for epoch in range(flags["epoch"]):
        if not early_stopping['stop']:
            # Update model parameter
            model.update_epoch(epoch + 1)
            # Display info
            print('-' * 55)
            print('EPOCH {}/{}'.format(epoch + 1, EPOCHS))
            print('-' * 55)

            # Training pass
            train_start = time.time()
            print('- training...')
            new_accuracies, new_losses, training_accuracy, training_loss = _train_epoch_fn(model=model,
                                                                                           ensamble=ensamble_models,
                                                                                           loader=training_loader,
                                                                                           criterion=criterion,
                                                                                           optimizer=optimizer,
                                                                                           device=device,
                                                                                           flags=flags)

            # Add new accuracies and losses
            training_accuracies += new_accuracies
            training_losses += new_losses

            del new_accuracies, new_losses
            gc.collect()

            # Evaluation pass
            valid_start = time.time()
            print('- validation...', flush=True)
            new_accuracies, new_losses, eval_accuracy, eval_loss = _eval_epoch_fn(model=model,
                                                                                  ensamble=ensamble_models,
                                                                                  loader=eval_loader,
                                                                                  criterion=criterion,
                                                                                  device=device,
                                                                                  flags=flags)

            # Add new accuracies and losses
            eval_accuracies += new_accuracies
            eval_losses += new_losses

            del new_accuracies, new_losses
            gc.collect()

            # Save weights
            if eval_loss < early_stopping['best']:
                torch.save(model.model_representation(), PATH_MODELS + "/" + flags["filename"])
                early_stopping['best'] = eval_loss
                early_stopping['no_improvement'] = 0
            else:
                early_stopping['no_improvement'] += 1
                if early_stopping['no_improvement'] == early_stopping['patience']:
                    early_stopping['stop'] = True

            # Display info
            print('- elapsed time | train = {:.2f} min, valid = {:.2f} min'.format(
                (valid_start - train_start) / 60, (time.time() - valid_start) / 60))
            print('- average loss | train = {:.6f}, valid = {:.6f}'.format(training_loss, eval_loss))
            print('- average accuracy | train = {:.6f}, valid = {:.6f}'.format(training_accuracy, eval_accuracy))
            print('-' * 55)
            print('')

            del training_loss, eval_loss, training_accuracy, eval_accuracy
            gc.collect()

            # Save training and evaluation values collected up to the current epoch
            save_model_info(training_losses, eval_losses, training_accuracies, eval_accuracies, flags["filename"])
            print(f"weight 1: {model.weight1.data.item()}, weight 2: {model.weight2.data.item()},"
                  f" weight 3: {model.weight3.data.item()}")

        else:
            print('- early stopping triggered ')
            break


if __name__ == "__main__":
    flags = parsing()
    _run(flags)
