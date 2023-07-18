import math

import pandas as pd
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from CONSTANTS import *
from dataset import ReviewDataset
from model import Roberta


def train(epoch, model, training_loader, optimizer, loss1, loss2, eos_token_id):
    training_loss = 0
    num_correct = 0
    num_training_steps = 0
    num_training_examples = 0

    lr_lambda = lambda num_training_steps: LEARNING_RATE + \
                                           ((PEAK_LEARNING_RATE - LEARNING_RATE) / WARMUP_STEPS) * num_training_steps \
        if num_training_steps <= WARMUP_STEPS \
        else (PEAK_LEARNING_RATE - LEARNING_RATE) * math.sqrt(WARMUP_STEPS) \
             * math.sqrt(num_training_steps) + LEARNING_RATE

    scheduler = LambdaLR(optimizer, lr_lambda)

    for i, data in tqdm(enumerate(training_loader, 0)):
        # Obs: each element of the dataset is a dictionary with ids, mask, token_type_ids, cls_target and
        # tokens_targets. Cls target is a tensor containing one element between 1 and 5, while tokens_targets is a
        # tensor containing the id of each masked token.
        ids = data['ids']  # .to(device, dtype=torch.int)
        mask = data['mask']  # .to(device, dtype=torch.int)
        token_type_ids = data['token_type_ids']  # .to(device, dtype=torch.int)
        cls_targets = data['cls_target']  # .to(device, dtype=torch.float)
        tokens_targets = data['tokens_targets']  # .to(device, dtype=torch.int)

        outputs = model(ids, mask, token_type_ids, eos_token_id)
        loss1_val = loss1(outputs['cls'], cls_targets)
        loss2_val = loss2(outputs['tokens'].view(-1, outputs['tokens'].size(-1)), tokens_targets.reshape(-1))
        loss = loss1_val + loss2_val
        training_loss += loss1_val.item() + loss2_val.item()

        # Accuracy on review prediction
        num_correct += torch.eq(torch.max(torch.softmax(outputs['cls'], 0).data, dim=1)[1], cls_targets).sum().item()

        # Increase number of training steps and number of examples processed in the current epoch
        num_training_steps += 1
        num_training_examples += cls_targets.size(0)

        if i % 4000 == 0:
            loss_step = training_loss / num_training_steps
            accu_step = (num_correct * 100) / num_training_examples
            print(f"Training Loss after {num_training_steps} steps: {loss_step}")
            print(f"Training Accuracy after {num_training_steps} steps: {accu_step}")

        # Set gradients to zero
        optimizer.zero_grad()
        # Compute gradient updates for learnable parameters
        loss.backward()
        # Update learnable parameters using grad attribute of the parameters, which has been updated by loss.backward()
        optimizer.step()
        # Compute new learning rate
        scheduler.step()

    print(f'Total Accuracy for Epoch {epoch}: {(num_correct * 100) / num_training_examples}')
    print(f"Average Training Loss Epoch: {training_loss / num_training_steps}")


if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataframe
    train_df = pd.read_json(PATH_DATASET, lines=True)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PATH_TOKENIZER)

    # Create dataset object and dataloader object
    training_set = ReviewDataset(train_df, tokenizer)

    # Create training loader
    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0}

    training_loader = DataLoader(training_set, **train_params)

    # Create model
    model = Roberta()
    # model.to(device)

    # Create the loss function and Adam optimizer
    loss1 = torch.nn.CrossEntropyLoss()
    loss2 = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    # Set model for training
    model.train()

    for epoch in range(EPOCHS):
        train(epoch, model, training_loader, optimizer, loss1, loss2, tokenizer.eos_token_id)

    # Save model
    torch.save(model.state_dict(), PATH_MODEL)
