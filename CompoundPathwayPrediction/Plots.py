import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

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
