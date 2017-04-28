#!/bin/bash
for i in {0..100}
do
python main.py --no-graph --cifar-svhn
done
