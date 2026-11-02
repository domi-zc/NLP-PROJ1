# NLP-PROJ1
NLP Project 1 (HSLU, HS26): Word Embeddings / Recurrent Neural Networks on **abductive
commonsense reasoning** ([ART](https://arxiv.org/abs/1908.05739), course version
[`amarfurt/art-nlp-hs26`](https://huggingface.co/datasets/amarfurt/art-nlp-hs26)).
Given two observations O1 (beginning of a story) and O2 (end), pick which of two hypotheses
h1/h2 better explains what happened in between.

## References and tools
- Task and dataset: Bhagavatula et al. (2020), [Abductive Commonsense Reasoning](https://arxiv.org/abs/1908.05739);
  course version of the data: [`amarfurt/art-nlp-hs26`](https://huggingface.co/datasets/amarfurt/art-nlp-hs26).
- Coding agent: Claude Code was used to help write the code and documentation.
- Pretrained embeddings: [GloVe 6B 300d](https://github.com/stanfordnlp/glove) (Pennington et al., 2014),
  downloaded from the official [stanfordnlp/glove](https://huggingface.co/stanfordnlp/glove) mirror.
- Libraries: PyTorch, PyTorch Lightning, Hugging Face `datasets`, NLTK (tokenization), Weights & Biases.
- Training setup (Lightning + `WandbLogger`) follows the course's experiment tracking exercise.

## Setup
The data is not part of the repository; everything goes into `data/`.

1. Dataset: download the three parquet files from the `anli/` folder of
   [`amarfurt/art-nlp-hs26`](https://huggingface.co/datasets/amarfurt/art-nlp-hs26/tree/main/anli)
   and put them, with their original names, into `data/art/`.
2. Python packages, embeddings and W&B login:
```bash
make setup
mkdir -p data && cd data
curl -L -o glove.6B.zip https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip
unzip glove.6B.zip glove.6B.300d.txt && rm glove.6B.zip && cd ..
uv run wandb login
```

## Running
All commands run from the project root.
```bash
make train-embedding [ARGS="..."]                   # one run of architecture 1 (e.g. ARGS="--learning_rate 1e-4 --dropout 0.5")
make train-rnn [ARGS="..."]                         # one run of architecture 2 (e.g. ARGS="--learning_rate 1e-4 --dropout 0.5")
make sweep SWEEP=embedding|rnn                      # create a W&B sweep, prints its id
make agent ID=<entity>/nlp-proj1-final/<sweep id>   # run the sweep
make seeds                                          # seed experiment
make test                                           # test set, final models only
```

## Approach
- **Preprocessing**: NLTK `word_tokenize`, lowercasing, no stemming/lemmatization, stopwords kept,
  only `.` `,` `/` `\` at the end of a text removed, unknown words → `<unk>` (unknown hyphenated words
  split first), no truncation, the 6 training examples with an empty hypothesis removed. Data behind
  each decision: [`notebooks/02_preprocessing.ipynb`](notebooks/02_preprocessing.ipynb).
- **Embeddings**: GloVe 6B 300d, frozen; vocabulary = all 400k GloVe words + `<pad>`, `<unk>`
  (zero vectors) and `<sep>` (fixed random vector).
- **Input/output**: every example becomes two candidates (O1, h1, O2) and (O1, h2, O2). The same
  network gives each candidate a score; the two scores are the logits of a softmax (cross-entropy),
  prediction = the higher score.
- **Architecture 1**: mean of the word vectors of O1, h and O2 separately (without `<pad>`/`<unk>`)
  → `[o1; h; o2]` (900) → classifier.
- **Architecture 2**: 2-layer unidirectional GRU over `O1 <sep> h <sep> O2` → last hidden state of
  the top layer → classifier.
- **Classifier** (both): Dropout → Linear → ReLU → Dropout → Linear(hidden, 1).
- **Training**: AdamW, batch size 128, constant learning rate, at most 100 epochs, early stopping on
  `val_acc` (patience 5), checkpoint of the epoch with the best `val_acc`.
- **Experiments** (W&B project `nlp-proj1-final`):
  1. Grid sweeps with seed 42, 48 runs per architecture: learning rate 1e-2/1e-3/1e-4/1e-5 ×
     dropout 0.2/0.5 × weight decay 0.01/0.1 × size (classifier resp. GRU 128/256/512).
  2. Seed experiment: the top 3 (arch 1) and top 4 (GRU) configurations again with seeds 3, 4, 13,
     21. Final configuration = best mean `val_acc` over the 5 seeds.
  3. Test: all 5 seeds of both final configurations, evaluated once.

## Results
Mean ± std over the seeds 3, 4, 13, 21, 42 (single runs in the W&B view).

| Model | Final configuration | val_acc | test_acc | Expected ([`EXPECTATIONS.md`](EXPECTATIONS.md)) |
|---|---|---|---|---|
| Always hypothesis 1 | – | 50.8% | 51.0% | ~50% |
| Arch 1 | lr 1e-3, dropout 0.2, wd 0.01, classifier 512 | 69.9 ± 0.5% | **51.7 ± 0.7%** | 55% |
| GRU | GRU 128, lr 1e-3, dropout 0.2, wd 0.01, classifier 256 | 71.3 ± 0.7% | **52.5 ± 0.6%** | 58% |

- The learning rate matters most (1e-3 best for both). The seed moves `val_acc` by about ±1 point,
  as much as most other hyperparameters: both sweep winners only had a lucky seed 42.
- Arch 1 underfits (train ≈ validation ≈ 70%). The GRU overfits (train ~96%, `val_loss` rises after
  a few epochs); early stopping keeps the epoch before.
- On test both are barely above always picking hypothesis 1; the GRU's +0.8 points are within the
  noise (standard error of a test accuracy ≈ 1.3 points).

The course test split is the [original ART](https://huggingface.co/datasets/allenai/art)
*dev* split, train and validation come from the original *train* split. According to the ART paper
([Bhagavatula et al. 2020](https://arxiv.org/abs/1908.05739), appendix A.2), the original train split
contains all crowdsourced hypothesis pairs, where the wrong hypothesis is often a small edit of the
right one, while dev and test keep only the pair per story that adversarial filtering against BERT
found hardest. Word-level cues that work on validation therefore do not carry over to test: both
models learned cues of the training data, not abductive reasoning.
