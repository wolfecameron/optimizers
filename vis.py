# contains helper methods for visualizing results of experiments

# necessary to fix issues with matplotlib on mac
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt

def plot_metrics(metric_dict, title='Training Losses', ylabel:str='Loss', colors:list=None):
    """plots losses in a matplotlib window

    Parameters:
    metric_dict: dict of form (exp_name, metric list)
    """

    if colors is None:
        colors = [u'b', u'g', u'r', u'c', u'm', u'y', u'k']
    assert len(colors) >= len(metric_dict), "Failure: too few colors were provided for plot_losses"

    # plot the results for each experiment with different color
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    for i, (exp_name, metrics) in enumerate(metric_dict.items()):
        epochs = [x for x in range(len(metrics))]
        plt.plot(epochs, metrics, label=exp_name, color=colors[i])
    plt.legend()
    plt.show()
