PyTorch implementation of PathNet: Evolution Channels Gradient Descent in Super Neural Networks.


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
-Learning rate is changed to 0.0001(paper) -> 0.01
-Path genotypes is not re-initialized at second task.


![Alt text](./imgs/generation_180.png?raw=true "Title")
## Result

![Alt text](./imgs/result.png?raw=true "Title")
