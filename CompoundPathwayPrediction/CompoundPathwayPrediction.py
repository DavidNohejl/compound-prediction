from PrepareData import *
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from skmultilearn.problem_transform import BinaryRelevance ,   LabelPowerset
from Evaluation import evaluate_classifier


#classifier = DecisionTreeClassifier(random_state=0,max_depth=5)

x_train,x_test,y_train,y_test = train_test_split(features,targets,test_size=0.33, random_state=42)

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y_train)
Y_test = mlb.fit_transform(y_test)

classifier = BinaryRelevance(classifier = DecisionTreeClassifier(random_state=0,max_depth=5))
classifier = classifier.fit(x_train, Y)

y_pred = classifier.predict(x_test).toarray()
y_pred_decoded = mlb.inverse_transform(y_pred)

evaluate_classifier(classifier,x_test,Y_test, y_pred,"BinaryRelevance DecisionTreeClassifier(random_state=0,max_depth=5)",x_train,Y,y_test)

classifier = LabelPowerset(classifier = DecisionTreeClassifier(random_state=0,max_depth=5))
classifier = classifier.fit(x_train, Y)
evaluate_classifier(classifier,x_test,Y_test, y_pred,"LabelPowerset DecisionTreeClassifier(random_state=0,max_depth=10)",x_train,y_train,Y)

#plot_confusion_matrix(classifier,x_test,Y_test.argmax(axis=1))
#plt.show()


# TODO: can't plot the tree anymore with BinaryRelevance thingy
#plt.figure()
#tree.plot_tree(classifier,filled=True)

#plt.savefig('tree.svg',format="svg")
#plt.show()


from skmultilearn.adapt import MLkNN
from sklearn.model_selection import GridSearchCV

parameters = {'k': range(1,5), 's': [0.5, 0.7, 1.0]}
score = 'f1_macro'

x_train = np.matrix(np.reshape(np.hstack(x_train),(-1,7)))
x_test = np.matrix(np.reshape(np.hstack(x_test),(-1,7)))

#clf = GridSearchCV(MLkNN(), parameters, scoring=score)
#clf.fit(x_train, Y)

#print (clf.best_params_, clf.best_score_)

classifier = MLkNN(k=1,s=0.5)

prediction = classifier.fit(x_train, Y).predict(x_test).toarray()

evaluate_classifier(classifier,x_test,Y_test,prediction,"MLkNN(k=1,s=0.5)",x_train,y_train,Y)


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)

classifier.fit(x_train, Y)
prediction = classifier.predict(x_test)

evaluate_classifier(classifier,x_test,Y_test,prediction, "RandomForestClassifier",x_train,y_train,Y)
#seems to work, accuracy 89.79

from sklearn.svm import SVC

classifier = BinaryRelevance(classifier=SVC(gamma=2, C=1))

print(x_train.shape, Y.shape)
classifier.fit(x_train, Y)
prediction = classifier.predict(x_test)

evaluate_classifier(classifier,x_test,Y_test,prediction, "BinaryRelevance SVC",x_train,y_train,Y)


from sklearn.neural_network import MLPClassifier

# default MLP, 100 hidden neurons, relu, adam, L2 penalty default=0.0001, const learning rate,default=0.001

classifier = MLPClassifier(random_state=1, max_iter=300).fit(x_train, Y)
prediction = classifier.predict(x_test)

evaluate_classifier(classifier,x_test,Y_test,prediction,"MLPClassifier",x_train,y_train,Y)

classifier = MLPClassifier(random_state=1, max_iter=300,hidden_layer_sizes=(50,50)).fit(x_train, Y)
prediction = classifier.predict(x_test)

evaluate_classifier(classifier,x_test,Y_test,prediction,"MLPClassifier 50-50",x_train,y_train,Y)

classifier = MLPClassifier(random_state=1, max_iter=300,hidden_layer_sizes=(50,100,50)).fit(x_train, Y)
prediction = classifier.predict(x_test)

evaluate_classifier(classifier,x_test,Y_test,prediction,"MLPClassifier 50-100-50",x_train,y_train,Y)

classifier = MLPClassifier(random_state=1, max_iter=300,hidden_layer_sizes=(1000)).fit(x_train, Y)
prediction = classifier.predict(x_test)

evaluate_classifier(classifier,x_test,Y_test,prediction,"MLPClassifier 1000",x_train,y_train,Y)


from sklearn.ensemble import AdaBoostClassifier
classifier = BinaryRelevance(classifier=AdaBoostClassifier(n_estimators=100, random_state=0)).fit(x_train, Y)
prediction = classifier.predict(x_test)

evaluate_classifier(classifier,x_test,Y_test,prediction,"BinaryRelevance AdaBoostClassifier",x_train,y_train,Y)


# from http://scikit.ml/modelselection.html#Estimating-hyper-parameter-k-for-embedded-classifiers
# not helping much?  overall acc: 85.96%

#from skmultilearn.problem_transform import ClassifierChain, LabelPowerset
#from sklearn.model_selection import GridSearchCV
#from sklearn.naive_bayes import GaussianNB
#from sklearn.ensemble import RandomForestClassifier
##pip install python-louvain
#from skmultilearn.cluster import NetworkXLabelGraphClusterer
#from skmultilearn.cluster import LabelCooccurrenceGraphBuilder
#from skmultilearn.ensemble import LabelSpacePartitioningClassifier

#from sklearn.svm import SVC

#parameters = {
#    'classifier': [LabelPowerset(), ClassifierChain()],
#    'classifier__classifier': [RandomForestClassifier()],
#    'classifier__classifier__n_estimators': [10, 20, 50],
#    'clusterer' : [
#        NetworkXLabelGraphClusterer(LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False), 'louvain'),
#        NetworkXLabelGraphClusterer(LabelCooccurrenceGraphBuilder(weighted=True, include_self_edges=False), 'lpa')
#    ]
#}

#clf = GridSearchCV(LabelSpacePartitioningClassifier(), parameters, scoring = 'f1_macro')
#clf.fit(x_train, Y)
#prediction = clf.predict(x_test)
#print (clf.best_params_, clf.best_score_)
#evaluate_classifier(clf.best_estimator_.classifier,x_test,Y_test,prediction,"Grid Search")
