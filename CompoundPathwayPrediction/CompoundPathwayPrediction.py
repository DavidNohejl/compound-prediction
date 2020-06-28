# pip install molmass

from molmass import Formula
import urllib.request
import time

f = Formula('C21H29N7O14P2')
print(f.mass)
print(f.isotope.mass)

import re
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MultiLabelBinarizer

class CompoundComposition:
    def __init__(self,H,C,N,O,P,S,Chl):
        self.H = H
        self.C = C
        self.N = N
        self.O = O
        self.P = P
        self.S = S
        self.Chl = Chl

    def features(self):
        return [self.H,self.C,self.N,self.N,self.P,self.S,self.Chl]

    def __str__(self):
          return "H:"+str(self.H)+ " C:"+str(self.C)+" N:"+str(self.N)+" O:"+str(self.O)+" P:"+str(self.P)+" S:"+str(self.S)+" Chl:"+str(self.Chl)

table = dict()

def compoundToComposition(compound):
    # https://stackoverflow.com/questions/2974362/parsing-a-chemical-formula
    element_pat = re.compile("([A-Z][a-z]?)(\d*)")
      
    h = c = n = o =p =s = chl = 0
    for (element_name, count) in element_pat.findall(compound):

        table.setdefault(element_name, 0)
        table[element_name] = table[element_name]+1
        if count == "":
            count = 1
        if element_name == 'H':
            h = count
        if element_name == 'C':
            c = count
        if element_name == 'N':
            n = count
        if element_name == 'O':
            o = count
        if element_name == 'P':
            p = count
        if element_name == 'S':
            s = count
        if element_name == 'Chl':
            chl = count

    return CompoundComposition(h,c,n,o,p,s,chl)


def loadCompoundPathwayMapping():
    my_file = open("compound_pathway.txt", "r")
    mapping = dict()
    for line in my_file:
        (pathway, compound) = line.strip().split('\t')
        mapping.setdefault(compound, [])
        mapping[compound].append(pathway)
    return mapping

def loadCompounds():
    my_file = open("compound.txt", "r")
    list = []
    for line in my_file:
        (compound, description) = line.strip().split('\t')
        list.append((compound, description))
    return list

def loadPathways():
    my_file = open("pathway.txt", "r")
    list = []
    for line in my_file:
        (compound, description) = line.strip().split('\t')
        list.append((compound, description))
    return list

def getAllCompounds(list):
    s = "+".join([i[0] for i in list])
    return s


def loadPathwayClasses(pathways):
     i = 0
     pathwayClassLookup = dict()
     for p in pathways:
        response = urllib.request.urlopen('http://rest.kegg.jp/get/'+p[0])
        txt =  response.read().decode('utf-8').split('\n')
        for line in txt:
            if line.startswith('CLASS'):
                pathway_class = line.replace('CLASS','').strip()
                print(str(i)+p[0]+' '+p[1]+' '+pathway_class)
                pathwayClassLookup[p[0]] = pathway_class    
                break
        #time.sleep(0.1)
        i = i+1
     return pathwayClassLookup
    
comp = compoundToComposition("C21H29N7O14P2")

print(comp)

compounds = loadCompounds()
getAllCompounds(compounds)
pathways = loadPathways()
mapping = loadCompoundPathwayMapping()
#classes = loadPathwayClasses(pathways)

def loadAllForumulas():
    i = 0
    compoundFormulaLookup = dict()
    for c in compounds:
        response = urllib.request.urlopen('http://rest.kegg.jp/get/'+c[0])
        txt =  response.read().decode('utf-8').split('\n')
        for line in txt:
            if line.startswith('FORMULA'):
                formula = line.split(' ')[-1]
                print(str(i)+c[0]+' '+c[1]+' '+formula)
                compoundFormulaLookup[c[0]] = formula    
                break
        #time.sleep(0.1)
        i = i+1
    import csv    
    w = csv.writer(open("compoundFormulaLookup.csv", "w"))
    for key, val in compoundFormulaLookup.items():
        w.writerow([key, val])

with open('compound_formula.txt','r') as inf:
    formulas = eval(inf.read())

with open('pathway_classes.txt','r') as inf:
    classes = eval(inf.read())


metabolism_classes = {key:value for (key,value) in classes.items() if value.startswith("Metabolism;")}

mlb = MultiLabelBinarizer()
bin_classes = mlb.fit_transform(metabolism_classes.values())
from sklearn import preprocessing
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

lb = preprocessing.LabelBinarizer()
lb.fit(list(metabolism_classes.values()))
encoded = lb.transform(list(metabolism_classes.values()))

clf = DecisionTreeClassifier(random_state=0,max_depth=3)

# features = compound formula
# target = pathway class 

relevant_formulas = {k:v for (k,v) in formulas.items() if k in mapping.keys() }


#pathways = pathways[]


def map_compound_to_pathway_classes(mapping, classes):
    #encoded_dict = {k:v for (k,v) in zip(list(metabolism_classes.keys()), encoded)}
    result = dict()
    lb = preprocessing.LabelBinarizer()
    lb.fit(list(metabolism_classes.values()))
    for k,v in mapping.items():
        pathway_classes = set([classes[c] for c in v if c in classes.keys()])
        if len(pathway_classes) > 0:
            result[k] = sum(lb.transform(list(pathway_classes)))
    return result

encoded_dict = map_compound_to_pathway_classes(mapping, classes)

relevant_formulas = {k:v for (k,v) in relevant_formulas.items() if k in encoded_dict.keys() }
composition_list = list(map(compoundToComposition,list(relevant_formulas.values())))

encoded_dict= {k:v for (k,v) in encoded_dict.items() if k in relevant_formulas.keys() }

table_sum = sum(table.values())
excluded = [x for x in table.keys() if table[x]/table_sum < 0.01 ]
# TODO: ummm protoze je to multiclass, tak to tahle nejde :D

included = [x for x in table.keys() if table[x]/table_sum >= 0.01 and not x=='R']

features = list(map(lambda x: x.features(),composition_list))

targets = [mapping[c] for c in list(relevant_formulas.keys())]
clf = clf.fit(features, list(encoded_dict.values()))

plt.figure()
tree.plot_tree(clf,filled=True)

plt.savefig('tree.svg',format="svg")
plt.show()