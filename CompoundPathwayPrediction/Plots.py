import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PrepareData import *
import itertools
from sklearn.metrics import confusion_matrix


def plot_distribution(data,labels,filename,xlabel, ylabel):
    
    sns.set_style('whitegrid')
    fig,ax = plt.subplots()
    x_pos = np.arange(len(labels))
    ax = sns.barplot(x=data,y=labels,orient='h')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.set_yticks(x_pos)
    ax.set_yticklabels(labels)
    ax.set_ylim(-1,len(labels))
    total_sum = sum(data)

    for i, p in enumerate(ax.patches):
        ax.annotate("%.2f (%.2f)%%" % (p.get_width(), p.get_width()/total_sum*100),
                    (p.get_x() +p.get_width(), p.get_y() -0.25),
                    xytext=(5, 10), textcoords='offset points')
    plt.savefig(filename,format='pdf')
    plt.show()

def plot_pca(features,targets,n):
    pca = PCA(n_components = 2)
    pca.fit(features)
    plot(pca)

    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 Component PCA', fontsize = 20)

    rgb = []  
    for i in range(len(labels)):
       rgb[i] = colorsys.hsv_to_rgb(i / 300., 1.0, 1.0)
    #colors = ['r', 'g', 'b']
    for target, color in zip(label,rgb):
        indicesToKeep = labels == target
        ax.scatter(features[indicesToKeep, 'principal component 1']
                   , features[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
    
from sklearn.metrics import roc_curve, auc

def plot_auc(classifier, y_true,y_score):

    from sklearn import datasets, metrics
    
    #for i in range(11):
     #   metrics.roc_curve(y_true[:,i],y_pred[:,i].toarray())

    #y_score = y_pred
    #y_test = y_true
    n_classes = len(set(metabolism_classes.values()))
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:,i],y_score[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    #for i in range(n_classes):
    #    plt.figure()
    #    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    #    plt.plot([0, 1], [0, 1], 'k--')
    #    plt.xlim([0.0, 1.0])
    #    plt.ylim([0.0, 1.05])
    #    plt.xlabel('False Positive Rate')
    #    plt.ylabel('True Positive Rate')
    #    plt.title('Receiver operating characteristic example')
    #    plt.legend(loc="lower right")
    #    plt.show()

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    #plt.plot(fpr["micro"], tpr["micro"],
    #         label='micro-average ROC curve (area = {0:0.2f})'
    #               ''.format(roc_auc["micro"]),
    #         color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    lw = 2
    colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()