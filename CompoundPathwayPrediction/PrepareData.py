import json
import urllib.request
import time
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing
from CompoundComposition import CompoundComposition, compoundToComposition

def loadCompoundPathwayMapping():
    my_file = open("./data/compound_pathway.txt", "r")
    mapping = dict()
    for line in my_file:
        (pathway, compound) = line.strip().split('\t')
        mapping.setdefault(compound, [])
        mapping[compound].append(pathway)
    return mapping

def loadCompounds():
    my_file = open("./data/compound.txt", "r")
    list = []
    for line in my_file:
        (compound, description) = line.strip().split('\t')
        list.append((compound, description))
    return list

def loadPathways():
    my_file = open("./data/pathway.txt", "r")
    list = []
    for line in my_file:
        (compound, description) = line.strip().split('\t')
        list.append((compound, description))
    return list

# used to pull data from KEGG
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
        i = i+1

     with open('./data/pathway_classes.txt', 'w') as fp:
        json.dump(pathwayClassLookup, fp)
     return pathwayClassLookup
    
# used to pull data from KEGG
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
        i = i+1
    
    with open('./data/compound_formula.txt', 'w') as fp:
        json.dump(compoundFormulaLookup, fp)



table = dict()

compounds = loadCompounds()
pathways = loadPathways()
mapping = loadCompoundPathwayMapping()

# loadPathwayClasses()
# loadAllForumulas()

with open('./data/compound_formula.txt','r') as inf:
    formulas = eval(inf.read())

with open('./data/pathway_classes.txt','r') as inf:
    classes = eval(inf.read())


metabolism_classes = {key:value for (key,value) in classes.items() if value.startswith("Metabolism;")}

# just to check, we exclude atoms with < 1% , and we are left exactly with H,C,N,O,P,S,Chl ...
table_sum = sum(table.values())

excluded = [x for x in table.keys() if table[x]/table_sum < 0.01 ]

included = [x for x in table.keys() if table[x]/table_sum >= 0.01 and not x=='R']

relevant_formulas = {k:v for (k,v) in formulas.items() if k in mapping.keys() }

def map_compound_to_pathway_classes(mapping, classes):
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
composition_list = [compoundToComposition(c,table) for c in relevant_formulas.values()]

encoded_dict= {k:v for (k,v) in encoded_dict.items() if k in relevant_formulas.keys() }

features = list(map(lambda x: x.features(),composition_list))

targets = list(encoded_dict.values())
