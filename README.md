# optimizers
implements various optimizers from scratch for analysis and comparison

# how to get the MNIST data
wget http://deeplearning.net/data/mnist/mnist.pkl.gz
mkdir data_files; mv mnist.pkl.gz ./data_files/mnist.pkl.gz

# how to get the CIFAR-10 data
download CIFAR-10 dataset from https://www.cs.toronto.edu/~kriz/cifar.html
place the compressed CIFAR-10 dataset into the data_files directory
>>> tar xvzf cifar-10-python.tar.gz

# to create the environment/install dependencies
# must first install anaconda for this to work!
>>> conda create -n optimization python=3.6 anaconda
>>> conda activate optimization
>>> pip install -r requirements.txt
