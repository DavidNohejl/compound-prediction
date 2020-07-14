from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
import sklearn.metrics as skm

from sklearn.metrics import confusion_matrix,plot_confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from PrepareData import metabolism_classes
from Plots import plot_auc
import numpy as np
from scipy.sparse import issparse

def evaluate_classifier(classifier,x_test,y_true,y_pred,name,x_train,y_train,y_test):
    # https://www.researchgate.net/publication/273859036_Multi-Label_Classification_An_Overview
    #print("accuracy: {}", accuracy_score(y_true,y_pred))
    #print("haming loss: {}", hamming_loss(y_true,y_pred))
    cm = multilabel_confusion_matrix(y_true,y_pred)
    mc  = list(set(metabolism_classes.values()))
    total = 0.0
    total_prec = 0.0
    total_rec = 0.0
    print(name)
    for i in range(len(cm)):
        accuracy = (cm[i][0,0]+cm[i][1,1])/sum(cm[i].ravel())
        precision = (cm[i][0,0])/(cm[i][0,0]+cm[i][0,1])
        recall = (cm[i][0,0])/(cm[i][0,0]+cm[i][1,0])
        total+=accuracy
        total_prec+=precision
        total_rec+=recall
        print("acc of class {}: {:.2%}".format(mc[i],accuracy))
        print("precision of class {}: {:.2%}".format(mc[i],precision))
        print("recall of class {}: {:.2%}".format(mc[i],recall))
    print("overall acc: {:.2%}".format(total/len(cm)))
    print("overall prec: {:.2%}".format(total_prec/len(cm)))
    print("overall rec: {:.2%}".format(total_rec/len(cm)))

    #y_score = classifier.predict_proba(np.matrix(x_test))
    #if issparse(y_score):
    #    y_score = y_score.toarray()
    #else:
    #   y_score = np.array(y_score)
    #plot_auc(classifier,y_true,y_score)
        #print(skm.classification_report(y_true,y_pred))
    #scores = cross_val_score(classifier, x_test, y_true, cv=10)
    #print(scores)

