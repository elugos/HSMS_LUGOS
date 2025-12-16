import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

import logging
logger = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s|%(levelname)s|%(message)s',
                    # filename='output.log', 
                    encoding='utf-8', 
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)



class TripletLossEmbeddingModel(nn.Module):
    """
    Implements the TripletLossEmbeddingModel.
    This model learns representations from input features by training an embedding layer with a triplet margin loss function.
    Training input are triplets of features for anchor, positive, negative samples.
    """
    def __init__(self, input_dim, embed_dim):
        super(TripletLossEmbeddingModel, self).__init__()
        # Create the shared embedding layer
        self.embedding = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        # Compute embeddings
        return F.normalize(self.embedding(x), p=2, dim=1)


class ContrastiveLossEmbeddingModel(nn.Module):
    """
    Implements the ContrastiveLossEmbeddingModel.
    Implementation is nearly identical to TripletLossEmbeddingModel
    """
    def __init__(self, input_dim, embed_dim):
        super(ContrastiveLossEmbeddingModel, self).__init__()
        # Create the shared embedding layer
        self.embedding = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        # Compute embeddings
        return F.normalize(self.embedding(x), p=2, dim=1)


class TripletDataset(Dataset):
    """
    Implement triplet dataset class.
    """
    def __init__(self, x_anchors, x_positives, x_negatives):
        assert x_anchors.shape == x_positives.shape == x_negatives.shape
        self.num_samples, self.input_dim = x_anchors.shape
        self.anchors = x_anchors
        self.positives = x_positives
        self.negatives = x_negatives
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            self.anchors[idx],
            self.positives[idx],
            self.negatives[idx],
        )
    

class ContrastiveDataset(Dataset):
    """
    Implement contrastive dataset class.
    """
    def __init__(self, x1, x2, labels):
        assert x1.shape == x2.shape
        assert len(labels) == len(x1)
        self.num_samples, self.input_dim = x1.shape
        self.x1 = x1
        self.x2 = x2
        self.labels = labels
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            self.x1[idx],
            self.x2[idx],
            self.labels[idx]
        )

class ContrastiveLoss(nn.Module):
    """
    Because PyTorch does not have a built-in contrastive loss function, we create one.
    """
    def __init__(self, margin=1.0, dist_func=None):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

        if dist_func is None:
            self.dist_func = F.pairwise_distance
        else:
            self.dist_func = dist_func

    def forward(self, output1, output2, label):
        distances = self.dist_func(output1, output2)
        loss = torch.mean(label * distances.pow(2) +
                          (1 - label) * F.relu(self.margin - distances).pow(2))
        return loss


def cosine_distance(x1, x2):
    """
    Implement a cosine_distance function as PyTorch does not have a built-in one.
    """
    x1 = F.normalize(x1, p=2, dim=1)
    x2 = F.normalize(x2, p=2, dim=1)
    cosine_sim = torch.sum(x1 * x2, dim=1)
    return 1 - cosine_sim  # convert similarity to distance


def train_contrastive_loss(dataset, embed_dim,
                       dist_func=None,
                       margin=1.0,
                       batch_size=256,
                       epochs=10,
                       lr=0.001
                       ):
    """
    Training loop for triplet margin loss.

    Args:
        dataset (ContrastiveLossDataset) : The training dataset.
        embed_dim (int) : The size of the embedding vectors.
        dist_func (callable) : If using custom distance function (default is Euclidean distance).
        margin (float) : The size of the margin learned (larger forces more separation between positive and negative).
        batch_size (int) : Training batch size.
        epochs (int) : Training epochs.
        lr (float) : The learning rate.

    Returns:
        model (ContrastiveLossEmbeddingModel) : The trained model.
        loss (np.array(float)) : The training loss for each epoch.
    """

    input_dim = dataset.input_dim
    model = ContrastiveLossEmbeddingModel(input_dim, embed_dim)

    # Create loss function module
    criterion = ContrastiveLoss(margin, dist_func)

    # Init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Training loop
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_history = np.zeros(epochs)
    for i_e, epoch in enumerate(range(epochs)):
        total_loss = 0
        for x1, x2, labels in dataloader:
            optimizer.zero_grad()
            # Forward pass
            x1_embed = model(x1)
            x2_embed = model(x2)
            # Compute loss
            loss = criterion(x1_embed, x2_embed, labels)
            # Backpropagation
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_history[i_e] = total_loss/len(dataloader)
        
        logging.debug(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    return model, loss_history
    
def train_triplet_loss(dataset, embed_dim,
                       dist_func=None,
                       margin=1.0,
                       batch_size=256,
                       epochs=10,
                       lr=0.001
                       ):
    """
    Training loop for triplet margin loss.

    Args:
        dataset (TripletDataset) : The training dataset.
        embed_dim (int) : The size of the embedding vectors.
        dist_func (callable) : If using custom distance function (default is Euclidean distance).
        margin (float) : The size of the margin learned (larger forces more separation between positive and negative).
        batch_size (int) : Training batch size.
        epochs (int) : Training epochs.
        lr (float) : The learning rate.

    Returns:
        model (TripletLossEmbeddingModel) : The trained model.
        loss (np.array(float)) : The training loss for each epoch.
    """

    input_dim = dataset.input_dim
    model = TripletLossEmbeddingModel(input_dim, embed_dim)

    if dist_func is None:
        criterion = nn.TripletMarginLoss(margin=margin, p=2)
    else:
        criterion = nn.TripletMarginWithDistanceLoss(margin=margin, 
                                                     distance_function=dist_func)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Training loop
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_history = np.zeros(epochs)
    for i_e, epoch in enumerate(range(epochs)):
        total_loss = 0
        for anchor, positive, negative in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            anchor_embed = model(anchor)
            positive_embed = model(positive)
            negative_embed = model(negative)
            
            # Compute loss
            loss = criterion(anchor_embed, positive_embed, negative_embed)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        loss_history[i_e] = total_loss
        
        logging.debug(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    return model, loss_history


if __name__ == "__main__":
    """Sample run"""

    # Example usage
    input_dim = 384  # Just like SBERT embeddings
    embed_dim = 64  # Chosen embedding dimension

    # Sample random set
    a = torch.randn(100, input_dim)
    pos = torch.randn(100, input_dim)
    neg = torch.randn(100, input_dim) + 10
    # Create a TripletDataset with the generated data.
    dataset = TripletDataset(a, pos, neg)
    model, loss = train_triplet_loss(dataset, embed_dim)
    logger.info(f"(Triplet) Trained model. Loss: {loss}")

    # Example usage for Contrastive Loss
    # Generate random "input vectors"
    x1 = torch.randn(300, input_dim)
    x2 = torch.randn(300, input_dim)
    margin = 1.0  # Choose a margin
    # Random labels: 0 if dissimilar, 1 if similar.
    labels = torch.randint(0, 2, (len(x1), ), dtype=torch.float32)

    logger.info(f"{x1.shape}, {x2.shape}, {labels.shape}")
    dataset = ContrastiveDataset(x1, x2, labels)
    model, loss = train_contrastive_loss(dataset, embed_dim, margin=margin, epochs=20)
    logger.info(f"Model trained!")

    # Get embedding for new inputs

    x_test = torch.randn(3, input_dim)  # 3 random input vectors
    x_test_emb = model(x_test)  # Get embeddings for each of the 3 inputs
    print(x_test_emb.shape) 
