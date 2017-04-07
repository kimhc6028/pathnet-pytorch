#!/usr/bin/env sh
# Original source code from caffe : https://github.com/BVLC/caffe/blob/master/data/cifar10/get_cifar10.sh
# This scripts downloads the CIFAR10 (binary version) data and unzips it.

if [ -d data/cifar ]; then
    echo "data directory already present"
else
    mkdir data/cifar    
fi

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

echo "Downloading..."

wget --no-check-certificate http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

echo "Unzipping..."


tar -xf cifar-10-python.tar.gz && rm -f cifar-10-python.tar.gz
mv cifar-10-batches-py/* ./data/cifar/ && rm -rf cifar-10-batches-py

# Creation is split out because leveldb sometimes causes segfault
# and needs to be re-created.

echo "Done."
