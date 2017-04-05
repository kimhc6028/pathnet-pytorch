PyTorch implementation of [PathNet: Evolution Channels Gradient Descent in Super Neural Networks](https://arxiv.org/abs/1701.08734). 
"It is a neural network algorithm that uses agents embedded in the neural network whose task is to discover which parts of the network to re-use for new tasks"

![Alt text](./imgs/generation_180.png?raw=true "Title")


Currently implemented MNIST task.

## Requirements

- Python 2.7
- [numpy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [networkx](https://networkx.github.io/)
- [python-mnist](https://pypi.python.org/pypi/python-mnist/)
- [pytorch](http://pytorch.org/)

## Usage
Install prerequisites:

	$ apt-get install python-numpy python-matplotlib

	$ pip install python-mnist networkx

And install pytorch: See http://pytorch.org/.

Run with command:

    $ python main.py
    
## Modifications
-Learning rate is changed from 0.0001(paper) to 0.01.
-Path genotypes is not re-initialized at second task.
-Generations are limited to 100 for each task.



## Result
Learning speed of second task was faster.
![Alt text](./imgs/result.png?raw=true "Title")
