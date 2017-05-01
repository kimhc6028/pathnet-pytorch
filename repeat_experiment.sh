#!/bin/bash
for i in {0..5}
do
python main.py --no-graph --cifar-svhn --cifar-first
done

for i in {0..5}
do
python main.py --no-graph --cifar-svhn
done
