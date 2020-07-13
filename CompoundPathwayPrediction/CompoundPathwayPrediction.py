from PrepareData import *
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from skmultilearn.problem_transform import BinaryRelevance    
from Evaluation import evaluate_classifier

classifier = BinaryRelevance(classifier = DecisionTreeClassifier(random_state=0,max_depth=5))
#classifier = DecisionTreeClassifier(random_state=0,max_depth=5)

x_train,x_test,y_train,y_test = train_test_split(features,targets,test_size=0.33, random_state=42)

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y_train)
Y_test = mlb.fit_transform(y_test)

classifier = classifier.fit(x_train, Y)

y_pred = classifier.predict(x_test)
y_pred_decoded = mlb.inverse_transform(y_pred)

evaluate_classifier(classifier,x_test,Y_test, y_pred)

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

clf = GridSearchCV(MLkNN(), parameters, scoring=score)
clf.fit(x_train, Y)

print (clf.best_params_, clf.best_score_)

classifier = MLkNN(k=1,s=0.5)

prediction = classifier.fit(x_train, Y).predict(x_test)

evaluate_classifier(classifier,x_test,Y_test,prediction)


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)

classifier.fit(x_train, Y)
prediction = classifier.predict(x_test)

evaluate_classifier(classifier,x_test,Y_test,prediction)
#seems to work, accuracy 89.79

from sklearn.svm import SVC

classifier = SVC(gamma=2, C=1)

print(x_train.shape, Y.shape)
classifier.fit(x_train, Y) #ValueError: bad input shape (3726, 11)
prediction = classifier.predict(x_test)

evaluate_classifier(classifier,x_test,Y_test,prediction)
