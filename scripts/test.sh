#!/usr/bin/env bash

for run in biupahg2 tg4nnhfa 24hvipno k9zpwvnw u2bpbvny; do
  uv run python -m src.evaluate checkpoints/$run-*/best-*.ckpt --wandb_run_id $run
done

for run in elbyeyfj xv942gpd zpzp150e ck7sqjpx 4kgu71tn; do
  uv run python -m src.evaluate checkpoints/$run-*/best-*.ckpt --wandb_run_id $run
done
