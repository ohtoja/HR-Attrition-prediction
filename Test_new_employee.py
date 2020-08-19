# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:53:59 2020

@author: ukijarmaoh
"""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
#from sklearn.model_selection import KFold

df = pd.read_excel('new_prediction_training_data.xlsx')

df.rename(columns={'attr.2years' : 'attrtwoyears'}, inplace=True)
#print(df.head())

departments = pd.get_dummies(df.Department)
departments = departments.drop("Other", axis=1)
df= df.drop('Department', axis=1)
df = df.join(departments)

nationalities = pd.get_dummies(df.Nationality1)
nationalities = nationalities.drop("Other", axis=1)
df= df.drop('Nationality1', axis=1)
df = df.join(nationalities)

target = df.attrtwoyears
features = df.drop('attrtwoyears', axis=1)

features = features.drop('Employee ID', axis=1)
print(features.columns)

#kf = KFold(n_splits=5)
#kf.get_n_splits(features)

#for train_index, test_index in kf.split(features):
     #print("TRAIN:", train_index, "TEST:", test_index)
#     features_train, features_test = features.iloc[train_index], features.iloc[test_index]
 #    target_train, target_test = target.iloc[train_index], target.iloc[test_index]


model = GradientBoostingClassifier(max_depth=8,learning_rate=0.1, n_estimators=100, min_samples_split=0.1,)
#model.fit(features_train, target_train)

results= cross_val_score(model,features,target,cv=5,scoring='accuracy')
precision = cross_val_score(model, features, target, cv=5, scoring='precision')
recall = cross_val_score(model, features, target, cv=5, scoring='recall')
print('accuracy', results)
print('precision' , precision)
print('recall', recall)
print(results.mean()*100.0, results.std()*100.0)

