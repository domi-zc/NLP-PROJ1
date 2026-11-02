import lightning as L
import torch
import torch.nn.functional as F

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from src.preprocessing import PAD_IDX, UNK_IDX


class Classifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(self.dropout(x)))
        return self.fc2(self.dropout(x)).squeeze(-1)


def mean_pool(e: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    mask = ((ids != PAD_IDX) & (ids != UNK_IDX)).unsqueeze(-1).float()
    return (e * mask).sum(1) / mask.sum(1).clamp(min=1)


class EmbeddingClassifier(nn.Module):
    def __init__(
        self,
        embeddings: torch.Tensor,
        classifier_hidden: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.embeddings = nn.Embedding.from_pretrained(
            embeddings, freeze=True, padding_idx=PAD_IDX
        )
        self.classifier = Classifier(
            3 * embeddings.shape[1], classifier_hidden, dropout
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        o1 = mean_pool(self.embeddings(batch["o1"]), batch["o1"])
        o2 = mean_pool(self.embeddings(batch["o2"]), batch["o2"])
        h = mean_pool(self.embeddings(batch["hyps"]), batch["hyps"])

        o1 = o1.repeat_interleave(2, dim=0)
        o2 = o2.repeat_interleave(2, dim=0)

        x = torch.cat([o1, h, o2], dim=-1)
        return self.classifier(x).reshape(-1, 2)


class RNNClassifier(nn.Module):
    def __init__(
        self,
        embeddings: torch.Tensor,
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        classifier_hidden: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.embeddings = nn.Embedding.from_pretrained(
            embeddings, freeze=True, padding_idx=PAD_IDX
        )
        self.rnn = nn.GRU(
            embeddings.shape[1],
            rnn_hidden,
            rnn_layers,
            dropout=dropout if rnn_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = Classifier(rnn_hidden, classifier_hidden, dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        e = self.embeddings(batch["stories"])  # (2B, L, 300)
        p = pack_padded_sequence(
            e, batch["story_lengths"].cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.rnn(p)
        h = h_n[-1]

        return self.classifier(h).reshape(-1, 2)


class ARTModule(L.LightningModule):
    def __init__(
        self,
        embeddings: torch.Tensor,
        architecture: str = "embedding",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        **model_kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["embeddings"])
        if architecture == "embedding":
            self.model = EmbeddingClassifier(embeddings, **model_kwargs)
        elif architecture == "rnn":
            self.model = RNNClassifier(embeddings, **model_kwargs)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(batch)

    def _step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        logits = self(batch)
        loss = F.cross_entropy(logits, batch["labels"])
        acc = (logits.argmax(-1) == batch["labels"]).float().mean()

        on_step = stage == "train"
        self.log(f"{stage}_loss", loss, on_step=on_step, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_acc", acc, on_step=on_step, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def on_save_checkpoint(self, checkpoint: dict):
        checkpoint["state_dict"].pop("model.embeddings.weight")

    def on_load_checkpoint(self, checkpoint: dict):
        weight = self.model.embeddings.weight
        checkpoint["state_dict"]["model.embeddings.weight"] = weight

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            trainable,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
