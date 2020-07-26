# Predicting Compound involvement in pathway

This work started as class project for NAIL107 Machine Learning in Bioinformatics class at Charles University of Prague. Inspired by [1], given chemical formula of compound, 
we tried to predict metabolic pathways in which this compounds is involved. But unlike the approach in the book, we formulated the problem as multi-label classification, 
that is, every compound can be involved in multiple pathways. 

Implementation is in python using sci-kit learn (https://scikit-learn.org/) and scikit-multilern (http://scikit.ml/) libraries. Data are from KEGG  database (https://www.kegg.jp/).

## Results

Overall accuracy/precision/recall was calculated as average accuracy/precision/recall over all classes.  

|Method                                                 |Overall accuracy  | Overall precision | Overall recall|
|-------------------------------------------------------|------------------|-------------------|---------------|
|DecisionTreeClassifier (LabelPowerset/BinaryRelevance) |89.71%|99.60%|89.94%|
|MLkNN|86.26%|93.05%|**91.00%**|
|RandomForestClassifier |89.75%|**99.99%**|89.75%|
|SVC (BinaryRelevance)|89.31%|98.15%|90.48%|
|MLPClassifier|89.76%|99.83%|89.83%|
|MLPClassifier 50-50|**89.81%**|99.86%|89.87%|
|MLPClassifier 50-100-50 (no convergence)|89.77%|99.44%|90.07%|
|MLPClassifier 1000|89.74%|99.60%|89.97%|
|AdaBoostClassifier (BinaryRelevance)|89.69%|99.88%|89.77%|

Measures per class (DecisionTreeClassifier):

|Class|Accuracy|Precision|Recall|
|-----|--------|---------|------|
|Carbohydrate |90.74%|100.00%|90.74%|
|Energy|82.84%|99.54%|83.02%|
|Lipid|98.20%|100.00%|98.20%|
|Nucleotide|77.51%|99.29%|77.65%|
|Amino acid|96.24%|100.00%|96.24%|
|Other amino acids|83.17%|99.93%|83.21%|
|Glycan|89.16%|100.00%|89.16%|
|Cofactors/vitamins|97.22%|100.00%|97.22%|
|Terpenoids|86.55%|99.69%|86.78%|
|Secondary metabolites|90.03%|100.00%|90.03%|
|Xenobiotics|96.30%|100.00%|96.30%|

Contrast that with what we belive to be current state of the art [2], which achieved accuracy over 95% for all classes using structure data and graph neural networks.

[1] Yang, Z. R. (2010). Machine learning approaches to bioinformatics (Vol. 4). World scientific.

[2] Mayank Baranwal, Abram Magner, Paolo Elvati, Jacob Saldinger, Angela Violi, Alfred O Hero, A deep learning architecture for metabolic pathway prediction, Bioinformatics, Volume 36, Issue 8, 15 April 2020, Pages 2547–2553, https://doi.org/10.1093/bioinformatics/btz954
