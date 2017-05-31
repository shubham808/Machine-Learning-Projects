import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
import matplotlib.pyplot as plt
from tester import dump_classifier_and_data
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','total_stock_value','salary','fraction_to_poi_email','expenses','shared_receipt_with_poi'] 
# You will need to use more features
###Got rid of features_list = [deferral_payments','loan_advances'] because most are zero
### Load the dictionary containing the dataset



with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

def dict_to_list(key, normalizer):
    new_list = []

    for i in data_dict:
        if data_dict[i][key] == "NaN" or data_dict[i][normalizer] == "NaN":
            new_list.append(0.)
        elif data_dict[i][key] >= 0:
            new_list.append(format(float(data_dict[i][key]) / float(data_dict[i][normalizer]),'.5f'))
    return new_list


### create two lists of new features
fraction_from_poi_email = dict_to_list("from_poi_to_this_person", "to_messages")
fraction_to_poi_email = dict_to_list("from_this_person_to_poi", "from_messages")

### insert new features into data_dict
count = 0
for i in data_dict:
    data_dict[i]["fraction_from_poi_email"] = fraction_from_poi_email[count]
    data_dict[i]["fraction_to_poi_email"] = fraction_to_poi_email[count]
    count += 1
outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers.append((key,int(val)))
outliers = sorted(outliers,key=lambda x:x[1],reverse=True)[:4]
outliers.append(('BHATNAGAR SANJAY',888))
for x in outliers:
    print(x)
    data_dict.pop(x[0],0)


### Task 2: Remove outliers --done
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
#features_list.append('bonus')
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
k=0
for i in features_list:
    print(k)
    print(i)
    k = k + 1

for i in range(0,len(features_list)-1):
    
    tmp =[]
    k=0
    for x in features:
        tmp.append(float(x[i]))
    tmp = StandardScaler().fit_transform(tmp)
    for x in features:
        x[i]=float(tmp[k])
        k = k + 1

scaler = StandardScaler()
#features_list = ['poi','salary','bonus','total_stock_value','total_payments','restricted_stock']

for i in range(1,len(features_list)-1):
    for x in features:
        plt.scatter(x[0],x[i])
    plt.ylabel(features_list[i+1])
    plt.xlabel("salary")
    plt.show()

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
pipeline = Pipeline(steps=[("clf",RandomForestClassifier())])

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB

#
param_grid = {"clf__criterion": ["gini", "entropy"],
              "clf__min_samples_split": [2, 10, 20],
              "clf__max_depth": [None, 2, 5, 10],
              "clf__min_samples_leaf": [1, 5, 10],
              "clf__max_leaf_nodes": [None, 5, 10, 20],
              }
# for sklearn version 0.16 or prior, the class_weight parameter value is 'auto'
clf = GridSearchCV(pipeline, param_grid)
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
#### stratified shuffle split cross validation. For more info: 
##### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
#clf = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=5,
#            max_features=None, max_leaf_nodes=20, min_impurity_split=1e-07,
#            min_samples_leaf=5, min_samples_split=10,
#            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
#            splitter='best')
    # Example starting point. Try investigating other evaluation techniques!
##
clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=2,
            max_features=None, max_leaf_nodes=20, min_impurity_split=1e-07,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

for i in range(1):        
    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)
    print("###############")
   
    #labels=np.array(labels)
    #print(clf.best_estimator_)
    #print(clf.best_score_)
    print(cross_val_score(clf,features,labels,cv=5))
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)