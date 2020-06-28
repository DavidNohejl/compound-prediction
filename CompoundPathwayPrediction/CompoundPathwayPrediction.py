from PrepareData import *

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss

mlb = MultiLabelBinarizer()
bin_classes = mlb.fit_transform(metabolism_classes.values())

lb = preprocessing.LabelBinarizer()
lb.fit(list(metabolism_classes.values()))
encoded = lb.transform(list(metabolism_classes.values()))

clf = DecisionTreeClassifier(random_state=0,max_depth=5)

x_train,x_test,y_train,y_test = train_test_split(features,targets,test_size=0.33, random_state=42)

clf = clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print(accuracy_score(y_test,y_pred))
# https://www.researchgate.net/publication/273859036_Multi-Label_Classification_An_Overview
print(hamming_loss(y_test,y_pred))

plt.figure()
tree.plot_tree(clf,filled=True)

plt.savefig('tree.svg',format="svg")
plt.show()