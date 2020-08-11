from collections import Counter
import matplotlib.pyplot as plt
plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
params = {'text.usetex': True,
          'font.size': 11,
          'font.family': 'lmodern'
          }
plt.rcParams.update(params)


import os
import numpy as np

class Set_analyst:
    def __init__(self, given_loaders=None, given_set=None):
        #self.l_per_class = Counter()
        self.given_loaders = given_loaders
        self.given_set = given_set

    def class_counter(self):
        if self.given_set is not None:
            self.l_per_class = Counter(np.array(self.given_set.data.y))
            self.l_per_class = [self.l_per_class]

            return self.l_per_class

        if self.given_loaders is not None:
            l_per_class=[]

            for given_loader in self.given_loaders:
                l_per_class_i = Counter()

                for data in given_loader:
                    a = np.array(data.y)
                    for i in a:
                        l_per_class_i[i] += 1

                l_per_class.append(l_per_class_i)

            return l_per_class



    def bar_plot(self, plotname, output_path):
        # saves a barplot of the given set (here either train or test)

        fig, ax = plt.subplots(figsize=(12, 7), edgecolor='k')


        ax.set_title('Samples per class', fontsize=20)
        plt.ylabel('Number of samples', fontsize=16)

        l_per_class_init = self.class_counter()

        for j, l_per_class in enumerate(l_per_class_init, start=0):

            two=False

            class_names = list(l_per_class.keys())
            class_counts = l_per_class.values()

            # Map from label index to label string
            if self.given_loaders is not None:
                ticks_a = [self.given_loaders[j].dataset.classmap[i] for i in class_names]
                ticks = [t.replace("_"," ") for t in ticks_a]
            if self.given_set is not None:
                ticks_a = [self.given_set.classmap[i] for i in class_names]
                ticks = [t.replace("_", " ") for t in ticks_a]


            plt.xlabel('Class label', fontsize=16)
            ind = np.arange(len(ticks))
            if j == 0:
                rec1 = ax.bar(ind, class_counts, 0.3, color='steelblue')
            if j==1:
                width = 0.4
                rec2 = ax.bar(ind+width, class_counts, 0.3, edgecolor='lightseagreen', color='None', linewidth=1.5)
                two=True

        if two:
            ax.legend((rec1[0], rec2[0]), ('Balanced dataset', 'Unbalanced dataset'), fontsize=16)
            class_names_i = class_names
            class_names = [k+0.2 for k in class_names_i]

        ax.set_xticks(class_names)
        ax.set_xticklabels(ticks, fontsize=12)

        filename = 'results_dataset_' + plotname + '.pgf'
        path = os.path.join(output_path, filename)
        plt.savefig(path)
        filename = 'results_dataset_' + plotname + '.pdf'
        path = os.path.join(output_path, filename)
        plt.savefig(path)

