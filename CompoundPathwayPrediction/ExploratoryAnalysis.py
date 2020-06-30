from PrepareData import * 

from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

from Plots import *

#plt.rcParams['figure.figsize'] = (10,10)
sum_targets = sum(targets)
labels = list(set(metabolism_classes.values()))
labels = list(map(lambda y: y.replace('Metabolism;',''),labels))


plot_distribution(sum_targets, labels,'exploratory_compound_classes.pdf',"Number of compounds","Pathway type")
