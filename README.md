# Optimizers
- implements various optimizers from scratch for analysis and comparison

# Get MNIST Data
> wget http://deeplearning.net/data/mnist/mnist.pkl.gz
> mkdir data_files; mv mnist.pkl.gz ./data_files/mnist.pkl.gz

# Get CIFAR-10 Data
- download CIFAR-10 dataset from https://www.cs.toronto.edu/~kriz/cifar.html
- place the compressed CIFAR-10 dataset into the data_files directory
> tar xvzf cifar-10-python.tar.gz

# Create Environment/Install dependencies
- must first install anaconda for this to work!
> conda create -n optimization python=3.6 anaconda
> conda activate optimization
> pip install -r requirements.txt

# File Descriptions
- get_data.py: contains helper methods for downloading/formatting data
- models.py: contains all model definitions (in PyTorch)
- optimizers.py: contains all optimizer definitions
- training.py: contains main training loop and training specifications
- vis.py: contains a few helper methods for visualizing training results
- requirements.txt: lists dependencies
- optimizers_report.pdf: contains the writeup for these experiments
