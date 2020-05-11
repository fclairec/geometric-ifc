from collections import Counter
import matplotlib.pyplot as plt
import os
import numpy as np

def set_analyst(set, setname=None):
    # saves a barplot of the given set (here either train or test)

    #   Length od dataset and each class for normalization
    l_per_class = Counter(np.array(set.data.y))
    class_names = list(l_per_class.keys())
    class_counts = l_per_class.values()

    # Map from label index to label string
    ticks = [set.classmap[i] for i in class_names]

    fig, ax = plt.subplots(figsize=(12, 7), edgecolor='k')
    ax.set_xticks(class_names)
    ax.set_xticklabels(ticks)
    ax.bar(class_names, class_counts, 0.3)
    ax.set_title('nb of samples per class')
    plt.ylabel('Nb of samples')
    plt.xlabel('class label')

    filename = 'results_dataset_' + setname + '.png'
    path = os.path.join("../out", filename)
    plt.savefig(path)