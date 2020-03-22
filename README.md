# optimizers
implements various optimizers from scratch for analysis and comparison

# to get the needed data
wget http://deeplearning.net/data/mnist/mnist.pkl.gz
mkdir data_files; mv mnist.pkl.gz ./data_files/mnist.pkl.gz

# to create the environment/install dependencies
# must first install anaconda for this to work!
>>> conda create -n optimization python=3.6 anaconda
>>> conda activate optimization
>>> pip install -r requirements.txt
