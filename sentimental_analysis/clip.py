import torch
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import pandas as pd
import numpy as np

from PIL import Image
from torchvision import transforms


BATCH_SIZE = 32
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10
TEMPERATURE = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentimentClassificationHead(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SentimentClassificationHead, self).__init__()
        self.fc = torch.nn.Linear(input_dim, num_classes)

    def forward(self, text_features):
        # Concatenate or combine text and image features
        logits = self.fc(text_features)
        return logits

classification_head = SentimentClassificationHead(input_dim=512, num_classes=2)
classification_head.to(device)
classification_optimizer = torch.optim.AdamW(classification_head.parameters(), lr=LEARNING_RATE)


# Step 1: Data Collection and Preprocessing
# You'll need to prepare your dataset and preprocess it into a format that can be used during training.

train_data = pd.read_csv("data/twitter-datasets/preprocessed/train_preprocessed.csv")
eval_data = pd.read_csv("data/twitter-datasets/preprocessed/test_preprocessed.csv")

IMG_PATH = "data/Twitter"

POSITIVE_IMAGES_IDX =[i for i in range(1, 5)]  # pos images idx (add sarcastic)
NEGATIVE_IMAGES_IDX = [i for i in range(68, 72)] # pos images idx
encode_map = {
    "NEGATIVE": NEGATIVE_IMAGES_IDX,
    "POSITIVE": POSITIVE_IMAGES_IDX
}
encode_target_map = {
    "NEGATIVE": 0,
    "POSITIVE": 1,
}

train_data["image"] = train_data["target"].map(lambda x: f"{IMG_PATH}/{np.random.choice(encode_map[x])}.png")
eval_data["image"] = eval_data["target"].map(lambda x: f"{IMG_PATH}/{np.random.choice(encode_map[x])}.png")

train_data["target"] = train_data["target"].map(encode_target_map)
eval_data["target"] = eval_data["target"].map(encode_target_map)


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")  # Pretrained model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

class CustomDataset(Dataset):
    def __init__(self, texts, images, labels, transform=None):
        self.texts = texts
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        image_path = self.images[idx]
        label = self.labels[idx]

        # Load and transform the image to tensor
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return text, image, label

# Define image transformation
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to the required input size of CLIP
    transforms.ToTensor(),
])


# Prepare your data (texts, images, and labels) and create DataLoader.
texts = train_data["text"].to_list()  # List of tweets
images = train_data["image"].to_list()  # List of image paths
labels = train_data["target"].to_list() # List of labels or captions
dataset = CustomDataset(texts, images, labels, transform=image_transform)
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

eval_texts = train_data["text"].to_list()  # List of tweets
eval_images = train_data["image"].to_list()  # List of image paths
eval_labels = train_data["target"].to_list() # List of labels or captions
eval_dataset = CustomDataset(eval_texts, eval_images, eval_labels, transform=image_transform)
val_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Step 4: Contrastive Learning
# Now, you'll define a contrastive loss and perform training.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

def contrastive_loss(image_features, text_features):
    # Define your contrastive loss here.
    # Calculate cosine similarity between image and text embeddings
    similarity_matrix = F.cosine_similarity(image_features.unsqueeze(1), text_features.unsqueeze(0), dim=-1)

    # Create positive mask (matches between images and texts)
    positive_mask = torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=similarity_matrix.device)

    # Calculate numerator for positive pairs (exponential of similarity divided by temperature)
    numerator = torch.exp(similarity_matrix / TEMPERATURE)

    # Calculate denominator for negative pairs (sum of exponential similarities)
    denominator = torch.exp(similarity_matrix / TEMPERATURE).sum(dim=1, keepdim=True)

    # Calculate contrastive loss (negative log likelihood of positive pairs divided by sum of negative pairs)
    contrastive_loss = -torch.log(numerator[positive_mask] / denominator).mean()

    return contrastive_loss

import torch
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

# Step 1: Data Collection and Preprocessing
# You'll need to prepare your dataset and preprocess it into a format that can be used during training.

# Step 2: Load the Pretrained CLIP Model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")  # Pretrained model
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Step 3: Prepare Data for Training
# Create a custom dataset class for your data.
# Step 3: Prepare Data for Training and Evaluation
# Create a custom dataset class for your data.
class CustomDataset(Dataset):
    def __init__(self, texts, images, labels, transform=None):
        self.texts = texts
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        image_path = self.images[idx]
        label = self.labels[idx]

        # Load and transform the image to tensor
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return text, image, label

# Define image transformation
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to the required input size of CLIP
    transforms.ToTensor(),
])

SAMPLES = 400000

# Prepare your data (texts, images, and labels) and create DataLoader.
texts = train_data["text"].to_list()[:SAMPLES]  # List of tweets
images = train_data["image"].to_list()[:SAMPLES] # List of image paths
labels = train_data["target"].to_list()[:SAMPLES] # List of labels or captions
dataset = CustomDataset(texts, images, labels, transform=image_transform)
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

eval_texts = train_data["text"].to_list()[:SAMPLES//10]  # List of tweets
eval_images = train_data["image"].to_list()[:SAMPLES//10]  # List of image paths
eval_labels = train_data["target"].to_list()[:SAMPLES//10] # List of labels or captions
eval_dataset = CustomDataset(eval_texts, eval_images, eval_labels, transform=image_transform)
val_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Step 4: Contrastive Learning
# Now, you'll define a contrastive loss and perform training.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

def contrastive_loss(image_features, text_features):
    # Define your contrastive loss here.
    # Calculate cosine similarity between image and text embeddings
    similarity_matrix = F.cosine_similarity(image_features.unsqueeze(1), text_features.unsqueeze(0), dim=-1)

    # Create positive mask (matches between images and texts)
    positive_mask = torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=similarity_matrix.device)

    # Calculate numerator for positive pairs (exponential of similarity divided by temperature)
    numerator = torch.exp(similarity_matrix / TEMPERATURE)

    # Calculate denominator for negative pairs (sum of exponential similarities)
    denominator = torch.exp(similarity_matrix / TEMPERATURE).sum(dim=1, keepdim=True)

    # Calculate contrastive loss (negative log likelihood of positive pairs divided by sum of negative pairs)
    contrastive_loss = -torch.log(numerator[positive_mask] / denominator).mean()

    return contrastive_loss



# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()

    correct_predictions = 0.0
    total_samples = 0
    for i, batch in enumerate(tqdm(train_dataloader)):
        batch_texts, batch_images, batch_labels = batch
        # Get image and text embeddings
        inputs = processor(text=batch_texts, images=batch_images, return_tensors="pt", padding=True)
        image_features = model.get_image_features(inputs["pixel_values"].to(device))
        text_features = model.get_text_features(inputs["input_ids"].to(device), inputs["attention_mask"].to(device))

        # Calculate classification loss
        classification_logits = classification_head(text_features)
        batch_labels = torch.tensor(batch_labels, device=device)

        loss_classification = F.cross_entropy(classification_logits, batch_labels)

        # Calculate contrastive loss and backpropagate
        loss_constrastive = contrastive_loss(image_features, text_features)

        # Calculate similarity score (cosine similarity) between image-text pairs
        similarity_scores = torch.matmul(image_features, text_features.t())

        loss = loss_constrastive + loss_classification
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        predicted_labels = torch.argmax(classification_logits, dim=1)
        correct_predictions += torch.sum(predicted_labels == batch_labels)
        total_samples += len(batch_labels)

        if i % 250 == 0:
          print("Contrastive loss ", loss_constrastive)
          print("Classification loss", loss_classification)
          print("Accuracy ", correct_predictions/total_samples)

    # Optionally perform evaluation on the validation dataset
    model.eval()
    correct_predictions = 0.0
    total_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader)):
            batch_texts, batch_images, batch_labels = batch
            # Get image and text embeddings for evaluation
            inputs = processor(text=batch_texts, images=batch_images, return_tensors="pt", padding=True)
            image_features = model.get_image_features(inputs["pixel_values"].to(device))
            text_features = model.get_text_features(inputs["input_ids"].to(device), inputs["attention_mask"].to(device))

            # Calculate classification loss
            classification_logits = classification_head(text_features)
            loss_classification = F.cross_entropy(classification_logits, batch_labels.to(device))

            batch_labels = torch.tensor(batch_labels, device=device)
            # Calculate similarity score (cosine similarity) between image-text pairs
            similarity_scores = torch.matmul(image_features, text_features.t())

            # Do something with the similarity scores for evaluation
            predicted_labels = torch.argmax(classification_logits, dim=1)

            # Calculate accuracy
            correct_predictions += torch.sum(predicted_labels == batch_labels)
            total_samples += len(batch_labels)

            if i % 250 == 0:
              print("Classification loss", loss_classification)
        print("Accuracy ", correct_predictions/total_samples)

# Step 5: Save the trained model
model.save_pretrained("trained_clip_model")

# Step 6: Fine-Tuning (Optional)
# After pretraining, you can fine-tune the model on a downstream task with labeled data.
