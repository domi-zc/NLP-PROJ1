.PHONY: help setup train-embedding train-rnn sweep agent seeds test

ARGS =

help:
	@echo 'make setup                                          install the packages'
	@echo 'make train-embedding [ARGS="..."]                   one run of architecture 1 (e.g. ARGS="--learning_rate 1e-4 --dropout 0.5")'
	@echo 'make train-rnn [ARGS="..."]                         one run of architecture 2 (e.g. ARGS="--learning_rate 1e-4 --dropout 0.5")'
	@echo 'make sweep SWEEP=embedding|rnn                      create a W&B sweep, prints its id'
	@echo 'make agent ID=<entity>/nlp-proj1-final/<sweep id>   run the sweep'
	@echo 'make seeds                                          seed experiment'
	@echo 'make test                                           test set, final models only'

setup:
	uv sync

train-embedding:
	uv run python -m src.train --architecture embedding $(ARGS)

train-rnn:
	uv run python -m src.train --architecture rnn $(ARGS)

sweep:
	uv run wandb sweep sweeps/$(SWEEP).yaml

agent:
	uv run wandb agent $(ID)

seeds:
	bash scripts/seeds.sh

test:
	bash scripts/test.sh
