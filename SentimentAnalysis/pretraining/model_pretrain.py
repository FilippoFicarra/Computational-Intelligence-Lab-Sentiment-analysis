import random

import pandas as pd
import torch
from alive_progress import alive_bar
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# import torch_xla
# import torch_xla.core.xla_model as xm

from CONSTANTS import *
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


if __name__ == "__main__":
    # DEVICE
    # device = xm.xla_device()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(PATH_TOKENIZER)

    # DATA FOR TRAINING AND EVALUATION

    # Load dataframe with dataset
    df = pd.read_json(PATH_DATASET, lines=True, dtype="overall int, reviewText string")

    # Group by overall
    grouped_df = df.groupby('overall')

    # Sample 20 % of dataset for training
    training_values = []
    indeces_to_drop_training = []
    for overall, reviews_indeces in grouped_df.groups.items():
        for ind in random.sample(reviews_indeces.tolist(), k=math.ceil(0.2 * len(reviews_indeces))):
            training_values.append((overall, df.iloc[ind, 1]))
            indeces_to_drop_training.append(ind)

    df_after_drop = df.drop(indeces_to_drop_training).reset_index(drop=True)
    grouped_df_after_drop = df_after_drop.groupby('overall')

    # Sample 1 % of dataset for validation
    val_values = []
    for overall, reviews_indeces in grouped_df_after_drop.groups.items():
        for ind in random.sample(reviews_indeces.tolist(), k=math.ceil(0.025 * len(reviews_indeces))):
            val_values.append((overall, df_after_drop.iloc[ind, 1]))

    # Define training dataframe
    df_training = pd.DataFrame(data=training_values, columns=["overall", "reviewText"])

    # Define validation dataframe
    df_eval = pd.DataFrame(data=val_values, columns=["overall", "reviewText"])

    # Create dataloader object for both training and validation
    training_loader = DataLoader(ReviewDataset(df_training, tokenizer), batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(ReviewDataset(df_eval, tokenizer), batch_size=TRAIN_BATCH_SIZE)

    # MODEL

    model = Roberta().to(device)

    # LOSS FUNCTIONS AND OPTIMIZER

    # Create the loss functions and Adam optimizer
    loss_cls = torch.nn.CrossEntropyLoss(reduction="sum")
    # loss_tokens_training = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    # VALUES FOR EARLY STOPPING

    # Create dictionary for early stopping and other variables
    early_stopping = {'best': 0, 'no_improvement': 0, 'patience': 2, 'stop': False}
    training_accuracies = []
    training_losses = []
    eval_accuracies = []
    eval_losses = []

    # SCALER

    # scaler = torch.cuda.amp.GradScaler()

    # EPOCHS LOOP

    for epoch in range(EPOCHS):
        if not early_stopping['stop']:
            # Set model for training
            model.train()

            # Define variables for epoch accuracy and loss
            training_loss = 0
            num_correct = 0
            num_training_steps = 0
            num_training_examples = 0

            # Initialize optimizer learning rate and define scheduler
            optimizer.param_groups[0]['lr'] = LEARNING_RATE
            scheduler = LinearLR(optimizer, start_factor=1, end_factor=0.01, total_iters=WARMUP_STEPS)

            # Train for the epoch
            with alive_bar(len(training_loader), force_tty=True, title=f"Epoch {epoch}") as bar:
                for i, data in enumerate(training_loader, 0):
                    # Set gradients to zero
                    optimizer.zero_grad()
                    # Obs: each element of the dataset is a dictionary with ids, mask, token_type_ids, cls_target and
                    # tokens_targets. Cls target is a tensor containing one element between 1 and 5,
                    # while tokens_targets is a tensor containing the id of each masked token.
                    ids = data['ids'].to(device, dtype=torch.long)
                    mask = data['mask'].to(device, dtype=torch.long)
                    cls_targets = data['cls_target'].to(device, dtype=torch.long)
                    # tokens_targets = twitter-data['tokens_targets'].to(device, dtype=torch.int)

                    outputs = model(ids, mask)
                    loss_cls_val = loss_cls(outputs, cls_targets)

                    # loss_tokens_val = loss_tokens_training(outputs['tokens'].view(-1, outputs['tokens'].size(-1)),
                    # tokens_targets.reshape(-1))
                    # loss = loss_cls_val  # + loss_tokens_val
                    training_loss += loss_cls_val.item()  # + loss_tokens_val.item()

                    # Accuracy on review prediction
                    num_correct += torch.eq(torch.max(torch.softmax(outputs, 0).data, dim=1)[1],
                                            cls_targets).sum().item()

                    # Increase number of training steps and number of examples processed in the current epoch
                    num_training_steps += 1
                    num_training_examples += cls_targets.size(0)

                    if i % 4000 == 0:
                        loss_step = training_loss / num_training_steps
                        accu_step = (num_correct * 100) / num_training_examples
                        print(f"Epoch {epoch} Training Accuracy after {num_training_steps} steps: {accu_step}")
                        print(f"Epoch {epoch} Training Loss after {num_training_steps} steps: {loss_step}")

                    # Compute gradient updates for learnable parameters
                    loss_cls_val.backward()
                    # Update learnable parameters using grad attribute of the parameters, which has been updated by
                    # loss.backward()
                    optimizer.step()
                    # Compute new learning rate
                    scheduler.step()
                    # xm step
                    # xm.mark_step()
                    # scaler.update()
                    # Update bar
                    bar()

            # Compute training measures for the epoch
            epoch_accuracy = (num_correct * 100) / num_training_examples
            epoch_loss = training_loss / num_training_steps
            training_accuracies.append(epoch_accuracy)
            training_losses.append(epoch_loss)
            print(f'Total Training Accuracy for Epoch {epoch}: {epoch_accuracy}')
            print(f"Average Training Loss for Epoch {epoch}: {epoch_loss}")

            # Evaluate model for early stopping
            model.eval()

            eval_loss = 0
            num_correct_eval = 0
            num_eval_examples = 0
            num_eval_steps = 0

            with torch.no_grad():
                for eval_batch in eval_loader:
                    eval_ids = eval_batch['ids'].to(device, dtype=torch.long)
                    eval_mask = eval_batch['mask'].to(device, dtype=torch.long)
                    eval_cls_targets = eval_batch['cls_target'].to(device, dtype=torch.long)

                    eval_outputs = model(eval_ids, eval_mask)
                    loss_eval_val = loss_cls(eval_outputs, eval_cls_targets)
                    eval_loss += loss_eval_val.item()

                    num_correct_eval += torch.eq(torch.max(torch.softmax(eval_outputs, 0).data, dim=1)[1],
                                                 eval_cls_targets).sum().item()

                    num_eval_steps += 1
                    num_eval_examples += eval_cls_targets.size(0)

            # Compute training measures for the epoch
            epoch_accuracy_eval = (num_correct_eval * 100) / num_eval_examples
            epoch_loss_eval = eval_loss / num_eval_steps
            eval_accuracies.append(epoch_accuracy_eval)
            eval_losses.append(epoch_loss_eval)
            print(f'Total Evaluation Accuracy for Epoch {epoch}: {epoch_accuracy_eval}')
            print(f"Average Evaluation Loss for Epoch {epoch}: {epoch_loss_eval}")

            # Early stopping
            if epoch_accuracy_eval > early_stopping['best']:
                early_stopping['best'] = epoch_accuracy_eval
            else:
                early_stopping['no_improvement'] += 1

            if early_stopping['no_improvement'] >= early_stopping['patience']:
                print("Early stopping triggered. No improvement in validation loss.")
                early_stopping['stop'] = True

    # Save model
    torch.save(model.roberta, PATH_MODEL)

    # Plot accuracies and losses
    plot_accuracy_and_loss(training_accuracies, training_losses, eval_accuracies, eval_losses)