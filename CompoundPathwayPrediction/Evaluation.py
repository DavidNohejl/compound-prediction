from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
import sklearn.metrics as skm

from sklearn.metrics import confusion_matrix,plot_confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix


def evaluate_classifier(classifier,x_test,y_true,y_pred):
    # https://www.researchgate.net/publication/273859036_Multi-Label_Classification_An_Overview
    #print("accuracy: {}", accuracy_score(y_true,y_pred))
    #print("haming loss: {}", hamming_loss(y_true,y_pred))
    cm = multilabel_confusion_matrix(y_true,y_pred)
    total = 0.0
    for i in range(len(cm)):
        accuracy = (cm[i][0,0]+cm[i][1,1])/sum(cm[i].ravel())
        total+=accuracy
        print("acc of class {}: {:.2%}".format(i,accuracy))
    print("overall acc: {:.2%}".format(total/len(cm)))
        #print(skm.classification_report(y_true,y_pred))
    #scores = cross_val_score(classifier, x_test, y_true, cv=10)
    #print(scores)

