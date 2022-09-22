
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Algorithms
from sklearn.model_selection import cross_val_predict, RandomizedSearchCV, KFold
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,recall_score,precision_score,accuracy_score,f1_score,roc_curve,roc_auc_score,auc,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC



cm1_data = pd.read_csv('./cm1.csv', encoding = "utf-8")
jm1_data = pd.read_csv('./jm1.csv', encoding = "utf-8")
kc1_class_data = pd.read_csv('./kc1_class_data.csv', encoding = "utf-8")
kc1_module_data= pd.read_csv('./kc1_method_data.csv', encoding = "utf-8")
pc1_data = pd.read_csv('./pc1.csv', encoding = "utf-8")
pd.set_option('display.max_colwidth', None)

train_data_list = [jm1_data,pc1_data,kc1_module_data]
train_data = pd.concat(train_data_list)

columns = train_data.select_dtypes(include=['Int64','float64','int32'])
def scale_features(data):
  transformer = preprocessing.Normalizer().fit(data.loc[:,columns.columns])
  data.loc[:,columns.columns] = transformer.transform(data.loc[:,columns.columns])

"""**CLASSIFIERS**"""

def classifier(train_data, label):
  random_forest = RandomForestClassifier(random_state=20)
  random_forest.fit(train_data,label)
  return random_forest
def second_classifier(train_data, label):
   random_forest = RandomForestClassifier(
                                          bootstrap=True,
                                          max_depth=40,
                                          max_features='sqrt',
                                          min_samples_leaf=35,
                                          min_samples_split=15,
                                          n_estimators=1200,
                                          class_weight = 'balanced',
                                          random_state=20,
                                          verbose=1
                                        )
   random_forest.fit(train_data,label)
   return random_forest
#Third classify for tuned hyper parametrs for KC1 class data
def third_classifier(train_data, label):
   random_forest = RandomForestClassifier(
                                          bootstrap=True,
                                          max_depth=20,
                                          max_features='sqrt',
                                          min_samples_leaf=15,
                                          min_samples_split=10,
                                          n_estimators=600,
                                          class_weight = 'balanced',
                                          random_state=20,
                                          verbose=1
                                        )
   random_forest.fit(train_data,label)
   return random_forest

#AUC Curve plotting method
def auc_roc_curve(fpr, tpr):
    auc_val = auc(fpr, tpr)
    plt.style.use('seaborn')
    plt.title('Area Under Curve (ROC)')
    plt.plot(fpr, tpr,color = 'blue', label = 'AUC = %0.2f' % auc_val)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show() 

# Performance evaluation method
def evaluate_model(label, prediction,classes):
  TN,FP,FN,TP = confusion_matrix(label,prediction).ravel()
  conf_matrix = confusion_matrix(label, prediction)
  dis = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=classes)
  dis.plot()
  plt.show()
  print('TN FP FN TP', (TN,FP,FN,TP))
  print('confusion matrix' ,'\n',conf_matrix)
  print('\n')
  print("Classification Report")
  print(classification_report(label,prediction,target_names=['non-defective', 'defective']))
  print('\n')
  print('accuracy_score', '\n', accuracy_score(label,prediction))
  print('recall_score', '\n', recall_score(label,prediction))
  print('precision_score', '\n', precision_score(label,prediction))
  print('f1_score', '\n', f1_score(label,prediction))

  # compute true positive rate and false positive rate
  false_positive_rate, true_positive_rate, thresholds = roc_curve(label,prediction)
  # plotting them against each other

  # plt.figure(figsize=(14, 7))
  auc_roc_curve(false_positive_rate, true_positive_rate)
  print('auc','-',roc_auc_score(label,prediction))

train_label = train_data['defective']
test_label = cm1_data['defective']
raw_train_data = train_data.drop('defective', axis=1)
transformed_train_data = raw_train_data.copy()
raw_test_data = cm1_data.drop('defective', axis=1)
transformed_test_data = raw_test_data.copy()

scale_features(transformed_train_data)
scale_features(transformed_test_data)

"""**Experiment 1a Training and Performance Evaluation**"""

rff_raw_data = classifier(raw_train_data,train_label)
rff_data_prediction = rff_raw_data.predict(raw_test_data)

evaluate_model(test_label, rff_data_prediction,rff_raw_data.classes_)

"""Experiment 1b Training and Performance Evaluation**"""
rfs_raw_data = second_classifier(raw_train_data,train_label)
rfs_data_prediction = rfs_raw_data.predict(raw_test_data)

evaluate_model(test_label, rfs_data_prediction,rfs_raw_data.classes_)

"""**Experiment 1c Training and Performance Evaluation**"""

rfs_transformed_data = second_classifier(transformed_train_data,train_label)
rfs_data_prediction = rfs_transformed_data.predict(transformed_test_data)

evaluate_model(test_label, rfs_data_prediction, rfs_transformed_data.classes_)

"""Obtain the features/metrics with the most importance"""

importances = pd.DataFrame({'feature':transformed_train_data.columns,'importance':np.round(rfs_transformed_data.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False)

"""Experiment 2 Training and Performance evaluation"""
correlated_features_to_drop = ['HalsteadEffort','TotalOperand','HalsteadDeliveredBug','HalsteadOperatorOperand','TotalOperator']
new_features = importances['importance']>0.003
new_features = importances[new_features.values]

new_columns = [features for features in new_features['feature'].values if features not in correlated_features_to_drop]
print(new_columns, len(new_columns))

new_train_data = pd.DataFrame(data=transformed_train_data,columns=new_columns)
new_test_data = pd.DataFrame(data=transformed_test_data,columns=new_columns)

new_train_data.info()

second_rf_model = second_classifier(new_train_data,train_label)
second_rf_model_prediction = second_rf_model.predict(new_test_data)

evaluate_model(test_label, second_rf_model_prediction,second_rf_model.classes_)

"""EXPERIMENT 3a"""

kc1_class_label = kc1_class_data['Defective']
kc1_class_train = kc1_class_data.drop('Defective', axis=1)

kFoldVal = KFold(shuffle=True,random_state=30, n_splits=8)
auc_values = []
tprs = []
fprs = []
i=0
colors = ['g','c','m','y','k','aquamarine','tab:purple','tab:brown','tab:pink','tab:gray']
colors_cn = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
plt.figure(figsize=(10,10))
plt.style.use('seaborn')
for train_index, test_index in kFoldVal.split(kc1_class_train):
  X_train, X_test = kc1_class_train.iloc[train_index,:],kc1_class_train.iloc[test_index,:]
  y_train, y_test = kc1_class_label[train_index], kc1_class_label[test_index]
  kc1_model = classifier(X_train,y_train)
  kc1_prediction = kc1_model.predict(X_test)
  false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,kc1_prediction)
  tprs.append(true_positive_rate)
  fprs.append(false_positive_rate)
  auc_value = auc(false_positive_rate, true_positive_rate)
  auc_values.append(auc_value)
  plt.plot(false_positive_rate,true_positive_rate, lw=1, alpha = 0.8,color = colors[i],label ='AUC For Fold %d = %0.2f'%(i+1,auc_value) )
  i+=1
mean_fprs = np.mean(fprs, axis=0)
mean_tprs = np.mean(tprs, axis=0)
mean_auc = np.mean(auc_values, axis = 0)
std_aucs = np.std(auc_values)
plt.title(label = 'ROC AUC Plot for KFold CV')
plt.plot(mean_fprs, mean_tprs, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_aucs),
         lw=3, alpha=.8)
plt.legend(loc = 'lower right',prop={'size':15})
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show() 
average_auc_value = sum(auc_values) / len(auc_values)

print('AUC of each fold - {}'.format(auc_values))
print('Average auc value: {}'.format(average_auc_value))

"""**Experiment 3b"""

kFoldVal = KFold(shuffle=True,random_state=30, n_splits=8)
auc_values = []
tprs = []
fprs = []
i=0
colors = ['g','c','m','y','k','aquamarine','tab:purple','tab:brown','tab:pink','tab:gray']
colors_cn = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
plt.figure(figsize=(10,10))
plt.style.use('seaborn')
for train_index, test_index in kFoldVal.split(kc1_class_train):
  X_train, X_test = kc1_class_train.iloc[train_index,:],kc1_class_train.iloc[test_index,:]
  y_train, y_test = kc1_class_label[train_index], kc1_class_label[test_index]
  kc1_model = third_classifier(X_train,y_train)
  kc1_prediction = kc1_model.predict(X_test)
  false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,kc1_prediction)
  tprs.append(true_positive_rate)
  fprs.append(false_positive_rate)
  auc_value = auc(false_positive_rate, true_positive_rate)
  auc_values.append(auc_value)
  plt.plot(false_positive_rate,true_positive_rate, lw=1, alpha = 0.8,color = colors[i],label ='AUC For Fold %d = %0.2f'%(i+1,auc_value) )
  i+=1
mean_fprs = np.mean(fprs, axis=0)
mean_tprs = np.mean(tprs, axis=0)
mean_auc = np.mean(auc_values, axis = 0)
std_aucs = np.std(auc_values)
plt.title(label = 'ROC AUC Plot for KFold CV')
plt.plot(mean_fprs, mean_tprs, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_aucs),
         lw=3, alpha=.8)
plt.legend(loc = 'lower right',prop={'size':15})
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show() 
average_auc_value = sum(auc_values) / len(auc_values)

print('AUC of each fold - {}'.format(auc_values))
print('Average auc value: {}'.format(average_auc_value))

"""EXPERIMENT 3c"""

# Features to drop from KC1_CLASS data based on Knowledge discovery
measures_of_encapsulation = [
    "PERCENT_PUB_DATA"
    "ACCESS_TO_PUB_DATA"
]
inheritance_metrics = [
    "COUPLING_BETWEEN_OBJECTS",
    "DEPTH"
]
irrelevant_features = [
   "PERCENT_PUB_DATA",
   "ACCESS_TO_PUB_DATA",
   "NUM_OF_CHILDREN",
   "DEP_ON_CHILD",
   "minLOC_BLANK",
   "minBRANCH_COUNT",
   "minLOC_CODE_AND_COMMENT",
   "minLOC_COMMENTS",
   "minCYCLOMATIC_COMPLEXITY",
   "minDESIGN_COMPLEXITY",
   "minESSENTIAL_COMPLEXITY",
   "minLOC_EXECUTABLE",
   "minHALSTEAD_CONTENT",
   "minHALSTEAD_DIFFICULTY",
   "minHALSTEAD_EFFORT",
   "minHALSTEAD_ERROR_EST",
   "minHALSTEAD_LENGTH",
   "minHALSTEAD_LEVEL",
   "minHALSTEAD_PROG_TIME",
   "minHALSTEAD_VOLUME",
   "minNUM_OPERANDS",
   "minNUM_OPERATORS",
   "minNUM_UNIQUE_OPERANDS",
   "minNUM_UNIQUE_OPERATORS",
   "minLOC_TOTAL",
   "maxDESIGN_COMPLEXITY",
   "maxLOC_BLANK",
   "maxBRANCH_COUNT",
   "maxLOC_CODE_AND_COMMENT",
   "maxLOC_COMMENTS",
   "maxCYCLOMATIC_COMPLEXITY",
   "maxESSENTIAL_COMPLEXITY",
   "maxLOC_EXECUTABLE",
   "maxHALSTEAD_CONTENT",
   "maxHALSTEAD_DIFFICULTY",
   "maxHALSTEAD_EFFORT",
   "maxHALSTEAD_ERROR_EST",
   "maxHALSTEAD_LENGTH",
   "maxHALSTEAD_LEVEL",
   "maxHALSTEAD_PROG_TIME",
   "maxHALSTEAD_VOLUME",
   "maxNUM_OPERANDS",
   "maxNUM_OPERATORS",
   "maxNUM_UNIQUE_OPERANDS",
   "maxNUM_UNIQUE_OPERATORS",
   "maxLOC_TOTAL",
   "sumLOC_BLANK",
   "sumBRANCH_COUNT",
   "sumDESIGN_COMPLEXITY",
   "sumLOC_CODE_AND_COMMENT",
   "sumLOC_COMMENTS",
   "sumCYCLOMATIC_COMPLEXITY",
   "sumESSENTIAL_COMPLEXITY",
   "sumLOC_EXECUTABLE",
   "sumHALSTEAD_CONTENT",
   "sumHALSTEAD_DIFFICULTY",
   "sumHALSTEAD_EFFORT",
   "sumHALSTEAD_ERROR_EST",
   "sumHALSTEAD_LENGTH",
   "sumHALSTEAD_LEVEL",
   "sumHALSTEAD_PROG_TIME",
   "sumHALSTEAD_VOLUME",
   "sumNUM_OPERANDS",
   "sumNUM_OPERATORS",
   "sumNUM_UNIQUE_OPERANDS",
   "sumNUM_UNIQUE_OPERATORS",
   "sumLOC_TOTAL"
]
new_kc1_data = kc1_class_data.drop(columns=irrelevant_features,axis=1)

kFoldVal = KFold(shuffle=True,random_state=30, n_splits=8)
auc_values = []
tprs = []
fprs = []
i=0
colors = ['g','c','m','y','k','aquamarine','tab:purple','tab:brown','tab:pink','tab:gray']
colors_cn = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
plt.figure(figsize=(10,10))
plt.style.use('seaborn')
for train_index, test_index in kFoldVal.split(new_kc1_data):
  X_train, X_test = new_kc1_data.iloc[train_index,:],new_kc1_data.iloc[test_index,:]
  y_train, y_test = kc1_class_label[train_index], kc1_class_label[test_index]
  kc1_model = classifier(X_train,y_train)
  kc1_prediction = kc1_model.predict(X_test)
  # evaluate_model(y_test, kc1_prediction,kc1_model.classes_)
  false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,kc1_prediction)
  tprs.append(true_positive_rate)
  fprs.append(false_positive_rate)
  auc_value = auc(false_positive_rate, true_positive_rate)
  auc_values.append(auc_value)
  plt.plot(false_positive_rate,true_positive_rate, lw=1, alpha = 0.8,color = colors[i],label ='AUC For Fold %d = %0.2f'%(i+1,auc_value) )
  i+=1
mean_fprs = np.mean(fprs, axis=0)
mean_tprs = np.mean(tprs, axis=0)
mean_auc = np.mean(auc_values, axis = 0)
std_aucs = np.std(auc_values)
plt.title(label = 'ROC AUC Plot for KFold CV')
plt.plot(mean_fprs, mean_tprs, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_aucs),
         lw=3, alpha=.8)
plt.legend(loc = 'lower right',prop={'size':15})
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show() 
average_auc_value = sum(auc_values) / len(auc_values)

print('AUC of each fold - {}'.format(auc_values))
print('Average auc value: {}'.format(average_auc_value))