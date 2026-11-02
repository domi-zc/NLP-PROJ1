import nltk
import numpy as np
import torch

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GLOVE_PATH = DATA_DIR / "glove.6B.300d.txt"
EMBEDDING_DIM = 300

PAD, UNK, SEP = "<pad>", "<unk>", "<sep>"
PAD_IDX, UNK_IDX, SEP_IDX = 0, 1, 2
END_CHARS = ".,/\\"

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def tokenize(text: str) -> list[str]:
    text = text.replace("’", "'").rstrip().rstrip(END_CHARS)
    return [t.lower() for t in nltk.word_tokenize(text)]


def hyphen_parts(token: str) -> list[str]:
    return [part for part in token.split("-") if part]


def load_glove() -> tuple[dict[str, int], torch.Tensor]:
    cache_path = DATA_DIR / "glove_cache.pt"
    if cache_path.exists():
        cache = torch.load(cache_path)
        return cache["vocab"], cache["embeddings"]

    vocab = {PAD: PAD_IDX, UNK: UNK_IDX, SEP: SEP_IDX}
    vectors = []
    with open(GLOVE_PATH, encoding="utf-8") as f:
        for line in f:
            word, *values = line.rstrip().split(" ")

            if word in vocab:
                continue
            vocab[word] = len(vocab)
            vectors.append(np.asarray(values, dtype=np.float32))
    glove = np.stack(vectors)

    zero = np.zeros(EMBEDDING_DIM, dtype=np.float32)

    rng = np.random.default_rng(42)
    sep = rng.normal(0, glove.std(), EMBEDDING_DIM).astype(np.float32)
    embeddings = torch.from_numpy(np.vstack([zero, zero, sep, glove]))

    torch.save({"vocab": vocab, "embeddings": embeddings}, cache_path)
    return vocab, embeddings


def encode(text: str, vocab: dict[str, int]) -> list[int]:
    ids = []
    for token in tokenize(text):
        parts = hyphen_parts(token)
        if token not in vocab and "-" in token and any(p in vocab for p in parts):
            ids += [vocab.get(p, UNK_IDX) for p in parts]
        else:
            ids.append(vocab.get(token, UNK_IDX))
    return ids
