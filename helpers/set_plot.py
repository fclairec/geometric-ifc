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
            print(np.unique(np.array(self.given_set.data.y)))
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


