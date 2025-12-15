import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

from embedding_models import cosine_distance, ContrastiveLoss

import logging
logger = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s|%(levelname)s|%(message)s',
                    # filename='output.log', 
                    encoding='utf-8', 
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)

# === Multi-task dataset for pairwise supervision ===
class MultiTaskPairDataset(Dataset):
    """
    Yields pairs (x1, x2) and a dict of task labels:
      labels = {
        "actor": float or 0/1,
        "action": float or 0/1,
        "geo": float or 0/1,
        "event": float or 0/1   # optional overall event similarity
      }
    All labels can be binary {0,1} or continuous in [0,1] (soft similarity).
    """
    def __init__(self, x1, x2, labels_dict):
        assert x1.shape == x2.shape
        self.num_samples, self.input_dim = x1.shape
        self.x1 = x1
        self.x2 = x2
        # Ensure all expected tasks exist; missing ones default to None
        self.tasks = ["actor", "action", "geo", "event"]
        self.labels_dict = {}
        for t in self.tasks:
            self.labels_dict[t] = labels_dict.get(t, None)

        # Basic checks for provided labels
        for t in self.tasks:
            if self.labels_dict[t] is not None:
                assert len(self.labels_dict[t]) == self.num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x1 = torch.tensor(self.x1[idx], dtype=torch.float32)
        x2 = torch.tensor(self.x2[idx], dtype=torch.float32)
        labels_out = {}
        for t in self.tasks:
            lbl = self.labels_dict[t]
            if lbl is None:
                labels_out[t] = None
            else:
                # detach/clone like your ContrastiveDataset style
                labels_out[t] = torch.tensor(
                    lbl[idx], dtype=torch.float32
                )
        return x1, x2, labels_out


# === Soft similarity loss for cosine-based supervision ===
class SoftCosineSimilarityLoss(nn.Module):
    """
    Supervise pairs using a target similarity y in [0,1].
    We compute cosine similarity in [-1,1], map to [0,1], and MSE against y.
      sim = cos(x1, x2)           # in [-1,1]
      sim_01 = (sim + 1) / 2      # in [0,1]
      loss = (sim_01 - y)^2
    Optionally include temperature to soften the cosine.
    """
    def __init__(self, temperature: float = 1.0):
        super(SoftCosineSimilarityLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2, y):
        # Normalize (already done in model heads, but safe here)
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        # Temperature-scaled cosine
        sim = torch.sum(z1 * z2, dim=1) / self.temperature  # [-1,1] approx
        sim_01 = (sim + 1.0) / 2.0                           # [0,1]
        return torch.mean((sim_01 - y) ** 2)


# === Multi-task embedding model ===
class MultiTaskContrastiveModel(nn.Module):
    """
    Shared trunk + per-task projection heads.
    Each head outputs a normalized embedding in its task-specific subspace.
    """
    def __init__(
        self,
        input_dim: int,
        base_dim: int = 256,
        task_embed_dim: int = 64,
        use_mlp: bool = True,
        dropout: float = 0.1,
    ):
        super(MultiTaskContrastiveModel, self).__init__()
        # Shared trunk: either linear or small MLP
        if use_mlp:
            self.trunk = nn.Sequential(
                nn.Linear(input_dim, base_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(base_dim, base_dim),
                nn.ReLU(),
            )
            trunk_out = base_dim
        else:
            self.trunk = nn.Linear(input_dim, base_dim)
            trunk_out = base_dim

        # Task-specific projection heads (all map to task_embed_dim)
        def head():
            return nn.Sequential(
                nn.Linear(trunk_out, task_embed_dim),
                nn.ReLU(),
                nn.Linear(task_embed_dim, task_embed_dim),
            )

        self.actor_head = head()
        self.action_head = head()
        self.geo_head = head()
        self.event_head = head()  # optional overall event similarity head

    def forward(self, x):
        h = self.trunk(x)
        out = {
            "actor": F.normalize(self.actor_head(h), p=2, dim=1),
            "action": F.normalize(self.action_head(h), p=2, dim=1),
            "geo": F.normalize(self.geo_head(h), p=2, dim=1),
            "event": F.normalize(self.event_head(h), p=2, dim=1),
        }
        return out


# === Training loop for multi-task pairwise contrastive/soft losses ===
def train_multitask_contrastive(
    dataset: MultiTaskPairDataset,
    task_embed_dim: int = 64,
    base_dim: int = 256,
    batch_size: int = 256,
    epochs: int = 10,
    lr: float = 1e-3,
    loss_cfg: dict = None,
    weights: dict = None,
):
    """
    Args:
      dataset: MultiTaskPairDataset
      task_embed_dim, base_dim: model dims
      loss_cfg: per-task loss config, e.g.
        {
          "actor": {"type": "binary_contrastive", "margin": 1.0},
          "action": {"type": "binary_contrastive", "margin": 1.0},
          "geo": {"type": "soft_cosine", "temperature": 1.0},
          "event": {"type": "soft_cosine", "temperature": 1.0}
        }
        Supported types: "binary_contrastive", "soft_cosine"
      weights: per-task scalars (static); e.g. {"actor":1.0,"action":1.0,"geo":1.0,"event":1.0}
    Returns:
      model, loss_history (np.array per-epoch)
    """
    input_dim = dataset.input_dim
    model = MultiTaskContrastiveModel(
        input_dim=input_dim,
        base_dim=base_dim,
        task_embed_dim=task_embed_dim,
    )

    # Default loss config
    default_cfg = {
        "actor": {"type": "binary_contrastive", "margin": 1.0},
        "action": {"type": "binary_contrastive", "margin": 1.0},
        "geo": {"type": "soft_cosine", "temperature": 1.0},
        "event": {"type": "soft_cosine", "temperature": 1.0},
    }
    if loss_cfg is None:
        loss_cfg = default_cfg
    else:
        # fill missing tasks from defaults
        for t in default_cfg:
            loss_cfg.setdefault(t, default_cfg[t])

    # Task weights
    if weights is None:
        weights = {t: 1.0 for t in ["actor", "action", "geo", "event"]}

    # Instantiate per-task criterion
    criterions = {}
    for t, cfg in loss_cfg.items():
        if cfg["type"] == "binary_contrastive":
            margin = cfg.get("margin", 1.0)
            # reuse your ContrastiveLoss (Euclidean by default) or cosine
            criterions[t] = ContrastiveLoss(margin=margin, dist_func=cosine_distance)
        elif cfg["type"] == "soft_cosine":
            temperature = cfg.get("temperature", 1.0)
            criterions[t] = SoftCosineSimilarityLoss(temperature=temperature)
        else:
            raise ValueError(f"Unknown loss type for task {t}: {cfg['type']}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_history = np.zeros(epochs)

    for i_e, epoch in enumerate(range(epochs)):
        total_loss = 0.0
        for x1, x2, labels_dict in dataloader:
            optimizer.zero_grad()
            z1 = model(x1)  # dict of embeddings per task
            z2 = model(x2)

            losses = []
            for t in ["actor", "action", "geo", "event"]:
                lbl = labels_dict[t]
                if lbl is None:
                    continue  # task not supervised in this batch
                # Expand to batch if needed
                if lbl.dim() == 0:
                    lbl = lbl.unsqueeze(0)

                crit = criterions[t]
                if isinstance(crit, ContrastiveLoss):
                    # Binary contrastive expects {0,1} labels
                    loss_t = crit(z1[t], z2[t], lbl)
                else:
                    # Soft cosine expects y in [0,1]
                    loss_t = crit(z1[t], z2[t], lbl)
                losses.append(weights[t] * loss_t)

            if len(losses) == 0:
                continue  # no supervised tasks present
            loss = torch.stack(losses).sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss_history[i_e] = total_loss/len(dataloader)
        logging.debug(f"Epoch {epoch+1}/{epochs}, MultiTask Loss: {total_loss/len(dataloader):.4f}")

    return model, loss_history



if __name__ == "__main__":
    input_dim = 384   # e.g., SBERT embeddings
    n = 500

    # Synthetic pair inputs
    x1 = torch.randn(n, input_dim)
    x2 = torch.randn(n, input_dim)

    # Make binary actor/action labels and soft geo/event labels
    actor_lbl  = torch.randint(0, 2, (n,), dtype=torch.float32)
    action_lbl = torch.randint(0, 2, (n,), dtype=torch.float32)

    # Suppose we have geographic distances in km; map to soft similarity
    # sim_geo = exp(-(d/sigma)^2). Here we just simulate distances.
    d_geo = torch.rand(n) * 2000.0  # 0..2000 km
    sigma = 500.0
    geo_lbl = torch.exp(-(d_geo / sigma) ** 2).to(torch.float32)  # in [0,1]

    # Overall event soft similarity (simulated)
    event_lbl = torch.rand(n).to(torch.float32)

    labels = {"actor": actor_lbl, "action": action_lbl, "geo": geo_lbl, "event": event_lbl}

    dataset = MultiTaskPairDataset(x1, x2, labels)

    loss_cfg = {
        "actor": {"type": "binary_contrastive", "margin": 1.0},
        "action": {"type": "binary_contrastive", "margin": 1.0},
        "geo": {"type": "soft_cosine", "temperature": 1.0},
        "event": {"type": "soft_cosine", "temperature": 1.0},
    }
    weights = {"actor": 1.0, "action": 1.0, "geo": 0.75, "event": 1.0}

    model, loss_hist = train_multitask_contrastive(
        dataset,
        task_embed_dim=64,
        base_dim=256,
        batch_size=128,
        epochs=20,
        lr=1e-3,
        loss_cfg=loss_cfg,
        weights=weights,
    )
    logger.info(f"(MultiTask) Trained model. Loss history shape: {loss_hist.shape}")
