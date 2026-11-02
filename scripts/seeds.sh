#!/usr/bin/env bash

for s in 3 4 13 21; do
  make train-embedding ARGS="--learning_rate 1e-3 --dropout 0.2 --weight_decay 0.01 --classifier_hidden 256 --seed $s --tags seeds"
  make train-embedding ARGS="--learning_rate 1e-3 --dropout 0.2 --weight_decay 0.1 --classifier_hidden 256 --seed $s --tags seeds"
  make train-embedding ARGS="--learning_rate 1e-3 --dropout 0.2 --weight_decay 0.01 --classifier_hidden 512 --seed $s --tags seeds"
done

for s in 3 4 13 21; do
  make train-rnn ARGS="--learning_rate 1e-3 --rnn_hidden 128 --dropout 0.5 --weight_decay 0.1 --seed $s --tags seeds"
  make train-rnn ARGS="--learning_rate 1e-3 --rnn_hidden 128 --dropout 0.2 --weight_decay 0.01 --seed $s --tags seeds"
  make train-rnn ARGS="--learning_rate 1e-3 --rnn_hidden 128 --dropout 0.5 --weight_decay 0.01 --seed $s --tags seeds"
  make train-rnn ARGS="--learning_rate 1e-3 --rnn_hidden 256 --dropout 0.5 --weight_decay 0.1 --seed $s --tags seeds"
done
