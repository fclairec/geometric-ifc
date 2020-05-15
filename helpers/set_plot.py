from collections import Counter
import matplotlib.pyplot as plt
import os
import numpy as np

class Set_analyst:
    def __init__(self, given_loader=None, given_set=None):
        self.l_per_class = Counter()
        self.given_loader = given_loader
        self.given_set = given_set

    def class_counter(self):
        if self.given_set is not None:
            self.l_per_class = Counter(np.array(self.given_set.data.y))

        if self.given_loader is not None:

            for data in self.given_loader:
                a = np.array(data.y)
                for i in a:
                    self.l_per_class[i] += 1

        return self.l_per_class


    def bar_plot(self, plotname):
        # saves a barplot of the given set (here either train or test)
        self.l_per_class = self.class_counter()
        class_names = list(self.l_per_class.keys())
        class_counts = self.l_per_class.values()

        # Map from label index to label string
        if self.given_loader is not None:
            ticks = [self.given_loader.dataset.classmap[i] for i in class_names]
        if self.given_set is not None:
            ticks = [self.given_set.classmap[i] for i in class_names]

        fig, ax = plt.subplots(figsize=(12, 7), edgecolor='k')
        ax.set_xticks(class_names)
        ax.set_xticklabels(ticks)
        ax.bar(class_names, class_counts, 0.3)
        ax.set_title('nb of samples per class')
        plt.ylabel('Nb of samples')
        plt.xlabel('class label')

        filename = 'results_dataset_' + plotname + '.png'
        path = os.path.join("../out", filename)
        plt.savefig(path)

