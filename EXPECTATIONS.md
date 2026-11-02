# Expectations

Expected best **test accuracy** of each model before running the experiments.

| Model | Expected test accuracy |
|---|---|
| Random / majority baseline | ~50% (two balanced classes) |
| Architecture 1: GloVe + 2-layer classifier | 55% |
| Architecture 2: GloVe + 2-layer LSTM/GRU + 2-layer classifier | 58% |

Reference points:
- Humans reach ~91% on ART, fine-tuned BERT-Large ~69% (Bhagavatula et al., 2020).
- The wrong hypotheses were made by changing only a few words of the right one, so word-level
  features can still help somewhat, but the task needs commonsense reasoning that neither
  averaged embeddings nor a small RNN trained on 17k examples really have.
