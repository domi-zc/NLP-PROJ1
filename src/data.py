import lightning as L
import torch

from pathlib import Path
from datasets import DatasetDict, load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from src.preprocessing import DATA_DIR, PAD_IDX, SEP_IDX, UNK_IDX, encode, load_glove

ART_DIR = DATA_DIR / "art"
SPLITS = ["train", "validation", "test"]
TEXT_COLUMNS = ["observation_1", "observation_2", "hypothesis_1", "hypothesis_2"]


def load_splits() -> DatasetDict:
    files = {s: str(ART_DIR / f"{s}-00000-of-00001.parquet") for s in SPLITS}
    missing = [f for f in files.values() if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset files (see README): {missing}")
    return load_dataset("parquet", data_files=files)


def has_letter(text: str) -> bool:
    return any(c.isalpha() for c in text)


class ARTDataset(Dataset):
    def __init__(self, split, vocab: dict[str, int]):
        self.examples = []
        for row in split:
            o1 = encode(row["observation_1"], vocab)
            o2 = encode(row["observation_2"], vocab)
            hyps = [
                encode(row["hypothesis_1"], vocab),
                encode(row["hypothesis_2"], vocab),
            ]
            self.examples.append(
                {
                    "o1": o1,
                    "o2": o2,
                    "hyps": hyps,
                    "stories": [o1 + [SEP_IDX] + h + [SEP_IDX] + o2 for h in hyps],
                    "label": row["label"] - 1,  # 1/2 -> class index 0/1
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def pad(seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    # an empty text gets one <unk>, the RNN needs at least one step
    tensors = [torch.tensor(s or [UNK_IDX]) for s in seqs]
    lengths = torch.tensor([len(t) for t in tensors])
    return pad_sequence(tensors, batch_first=True, padding_value=PAD_IDX), lengths


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    o1, _ = pad([ex["o1"] for ex in batch])
    o2, _ = pad([ex["o2"] for ex in batch])
    hyps, _ = pad([h for ex in batch for h in ex["hyps"]])
    stories, story_lengths = pad([s for ex in batch for s in ex["stories"]])
    return {
        "o1": o1,  # (B, L)
        "o2": o2,  # (B, L)
        "hyps": hyps,  # (2B, L)
        "stories": stories,  # (2B, L)
        "story_lengths": story_lengths,  # (2B,)
        "labels": torch.tensor([ex["label"] for ex in batch]),  # (B,)
    }


class ARTDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 128):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: str | None = None):
        if hasattr(self, "train_set"):
            return
        splits = load_splits()
        self.vocab, self.embeddings = load_glove()

        train = splits["train"]
        keep = [
            i
            for i, row in enumerate(train)
            if has_letter(row["hypothesis_1"]) and has_letter(row["hypothesis_2"])
        ]

        self.train_set = ARTDataset(train.select(keep), self.vocab)
        self.val_set = ARTDataset(splits["validation"], self.vocab)
        self.test_set = ARTDataset(splits["test"], self.vocab)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True, collate_fn=collate
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=256, collate_fn=collate)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=256, collate_fn=collate)
