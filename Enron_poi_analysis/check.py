#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
import matplotlib.pyplot as plt
from tester import dump_classifier_and_data, test_classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest,SelectPercentile
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.metrics import recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier

def get_nan_counts(dictionary):
    '''
    converts 'NaN' string to np.nan returning a pandas
    dataframe of each feature and it's corresponding
    percent null values (nan)
    '''
    my_df = pd.DataFrame(dictionary).transpose()
    nan_counts_dict = {}
    for column in my_df.columns:
        my_df[column] = my_df[column].replace('NaN',np.nan)
        nan_counts = my_df[column].isnull().sum()
        nan_counts_dict[column] = round(float(nan_counts)/float(len(my_df[column])) * 100,1)
    df = pd.DataFrame(nan_counts_dict,index = ['percent_nan']).transpose()
    df.reset_index(level=0,inplace=True)
    df = df.rename(columns = {'index':'feature'})
    return df

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments','total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'other', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock', 'director_fees', 'shared_receipt_with_poi']

# features_list = ['poi','salary','bonus','total_stock_value','total_payments','exercised_stock_options'] # You will need to use more features
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers.append((key,int(val)))

outliers = sorted(outliers,key=lambda x:x[1],reverse=True)[:4]
outliers.append(('BHATNAGAR SANJAY', 888))
for x in outliers:
    print(x)
    data_dict.pop(x[0],0)



### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
#
new_list=[]

def dict_to_list(key,normalizer):
    new_list=[]

    for i in data_dict:
        if data_dict[i][key]=="NaN" or data_dict[i][normalizer]=="NaN":
            new_list.append(0.)
        elif data_dict[i][key]>=0:
            new_list.append(float(data_dict[i][key])/float(data_dict[i][normalizer]))
    return new_list

fraction_from_poi_email=dict_to_list("from_poi_to_this_person","to_messages")
fraction_to_poi_email=dict_to_list("from_this_person_to_poi","from_messages")


count = 0
for i in data_dict:
    data_dict[i]["fraction_from_poi_email"] = fraction_from_poi_email[count]
    data_dict[i]["fraction_to_poi_email"] = fraction_to_poi_email[count]
    count += 1

my_dataset = data_dict

features_list = ['poi','total_stock_value','fraction_to_poi_email','expenses','shared_receipt_with_poi']

#features_list.append('bonus')
### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)

labels, features = targetFeatureSplit(data)


k=10
k_best = SelectKBest(k=2)
k_best.fit(features, labels)
scores = k_best.scores_
print(scores)
# pairs = zip(features_list[1:], scores)
#
# # combined scores and features into a pandas dataframe then sort
# k_best_features = pd.DataFrame(pairs)
# k_best_features = k_best_features.sort('score', ascending=False)
#
# # merge with null counts
# df_nan_counts = get_nan_counts(my_dataset)
# k_best_features = pd.merge(k_best_features, df_nan_counts, on='feature')
#
# # eliminate infinite values
# k_best_features = k_best_features[np.isinf(k_best_features.score) == False]
# print('Feature Selection by k_best_features\n')
# print("{0} best features in descending order: {1}\n".format(k, k_best_features.feature.values[:k]))
# print('{0}\n'.format(k_best_features[:k]))

#scaling
for i in range(0,len(features_list)-1):
    tmp =[]
    k=0
    for x in features:
        tmp.append(float(x[i]))
    tmp = StandardScaler().fit_transform(tmp)
    for x in features:
        x[i]=tmp[k]
        k = k + 1

#features_list = ['poi','salary','bonus','total_stock_value','total_payments','restricted_stock']
for x in features:
    plt.scatter(x[0],x[-1])

plt.xlabel("salary")
plt.show()

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
clf = SVC(C=10000.0, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=True,
          probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False,
          max_iter=-1, decision_function_shape=None, random_state=None)
# clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
#             max_features=None, max_leaf_nodes=20, min_impurity_split=1e-07,
#             min_samples_leaf=1, min_samples_split=5,
#             min_weight_fraction_leaf=0.0, presort=False, random_state=None,
#             splitter='best')
# clf = AdaBoostClassifier()
# import pickle
# with open('best_clf_pipe.pkl', 'rb') as f:
#     u = pickle._Unpickler(f)
#     u.encoding = 'latin1'
#     p = u.load()
test_classifier(clf,my_dataset,features_list)
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

#
# clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=50, max_features='auto', max_leaf_nodes=None,
#             min_impurity_split=1e-07, min_samples_leaf=3,
#             min_samples_split=2, min_weight_fraction_leaf=0.0,
#             n_estimators=20, n_jobs=1, oob_score=False, random_state=None,
#             verbose=0, warm_start=False)
# clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
# print(clf.feature_importances_)



print(recall_score(labels_test, pred))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)