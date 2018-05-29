# import statements
# Open below...

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import scipy as sp
import sklearn
import seaborn as sns
import Data_Cleaning as dc
import warnings

warnings.filterwarnings('ignore')

from sklearn import svm, tree, linear_model, neighbors,naive_bayes, ensemble, discriminant_analysis, gaussian_process
import xgboost
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics



# Read the datasets and Copy as values
data_raw = pd.read_csv("Databases/train.csv")
db_train = data_raw.copy()
db_test = pd.read_csv("Databases/test.csv").copy()

# Pass by reference to clean both databases at same time
target = db_train['Survived']
full_data = db_train.drop('Survived', axis=1).append(db_test, ignore_index=True)

# Check different categories to clean/complete
# print(db_train.columns.values)
# ['PassengerId' 'Survived' 'Pclass' 'Name' 'Sex' 'Age' 'SibSp' 'Parch' 'Ticket' 'Fare' 'Cabin' 'Embarked']

# PassengerId doesn't had any information so it will be dropped
full_data.drop('PassengerId', axis=1, inplace=True)

# Call function from the other py file for data cleaning
clean_data = dc.clean_data(full_data)

# ----------------------------------------------------------------------------------------------------------------------
# With our cleaned data now we need to create ordinal categories to look for correlations that will help us decide
# which features to keep in the model
# Using simple label encoding it is important that our categories are organized according to survival rates so we
# know how to interpret the correlation matrix
# (Try also dummies to see if model improves) - One Hot Encoder

# First divide our clean data and attach the target Series
db_train, db_test = pd.DataFrame(target).join(clean_data.loc[:target.size, :]), clean_data.loc[target.size:, :]
train = db_train.copy(); test = db_test.copy()
datasets = [db_train, db_test]


# Create dummies
db_train_dummy = pd.get_dummies(db_train[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                                          'Title', 'Family', 'Cat Family', 'IsAlone', 'Cat Age', 'Fare per Person',
                                          'Cat Fare']])
db_test_dummy = pd.get_dummies(db_test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'Family',
                                        'Cat Family', 'IsAlone', 'Cat Age', 'Fare per Person', 'Cat Fare']])
datasets_dummy = [db_train_dummy, db_test_dummy]

# Simply Labeled
# Looking at the survival means by group aggregate it is possible to see
# that Pclass and Embarked are already ordered correctly
# We then need to order the remaining categories
cat = ['Embarked', 'Title', 'Cat Family', 'Cat Age', 'Cat Fare', "Sex"]
cat_dict = {}
for c in cat:
	tmp = db_train['Survived'].groupby(db_train[c]).mean().to_dict()
	keys = sorted(tmp, key=tmp.__getitem__, reverse=True)
	cat_dict[c] = {k: keys.index(k) for k in keys}
	
	for dataset in datasets:
		dataset[c + '_Code'] = dataset[c].replace({k: keys.index(k) for k in keys})
		
# Finally we drop the features that don't matter for us:
# Other tests: drop less significant features
col_drop = ["Name", "Sex", "Ticket", "Fare", "Cabin", "Embarked", "Title", "Cat Family", "Cat Age", "Cat Fare"]
drop = ['SibSp', 'Parch', 'Family', 'IsAlone', 'Fare per Person', 'Age']
for dataset in datasets:
	dataset.drop(col_drop, axis=1, inplace=True)
	dataset.drop(drop, axis=1, inplace=True)


# One hot Encoding - dummies
drop = ['SibSp', 'Parch', 'Family', 'IsAlone', 'Fare per Person', 'Age', 'Fare']
for dataset in datasets_dummy:
	dataset.drop(drop, axis=1, inplace=True)

'''
# ---------------------------------------------------------------------------------------------------------------------
# Data Plotting and Analysis

fig, saxis = plt.subplots(1, 3,figsize=(16,12))
sns.pointplot(x = 'Cat Fare_Code', y = 'Survived',  data=db_train, ax = saxis[0])
sns.pointplot(x = 'Cat Age_Code', y = 'Survived',  data=db_train, ax = saxis[1])
sns.pointplot(x = 'Cat Family_Code', y = 'Survived', data=db_train, ax = saxis[2])
plt.show()


# correlation heatmap of dataset
def correlation_heatmap(df):
	_, ax = plt.subplots(figsize=(14, 12))
	colormap = sns.diverging_palette(220, 10, as_cmap=True)
	
	_ = sns.heatmap(df.corr(), square=True, cbar_kws={'shrink': .9}, ax=ax, annot=True, linewidths=0.01,
		vmax=1.0, linecolor='white', annot_kws={'fontsize': 8})
	
	plt.title('Pearson Correlation of Features', y=1.05, size=15)

 
#correlation_heatmap(db_train)
#plt.show()

# ---------------------------------------------------------------------------------------------------------------------
# Machine Learning

#

# Machine Learning Algorithm (MLA) Selection and Initialization
MLA = [# Ensemble Methods
	ensemble.AdaBoostClassifier(),
	ensemble.BaggingClassifier(),
	ensemble.ExtraTreesClassifier(),
	ensemble.GradientBoostingClassifier(),
	ensemble.RandomForestClassifier(),
	
	# Gaussian Processes
	gaussian_process.GaussianProcessClassifier(),
	
	# GLM
	linear_model.LogisticRegressionCV(),
	linear_model.PassiveAggressiveClassifier(),
	linear_model.RidgeClassifierCV(),
	linear_model.SGDClassifier(),
	linear_model.Perceptron(),
	
	# Navies Bayes
	naive_bayes.BernoulliNB(),
	naive_bayes.GaussianNB(),
	
	# Nearest Neighbor
	neighbors.KNeighborsClassifier(),
	
	# SVM
	svm.SVC(probability=True),
	svm.NuSVC(probability=True),
	svm.LinearSVC(),
	
	# Trees
	tree.DecisionTreeClassifier(),
	tree.ExtraTreeClassifier(),
	
	# Discriminant Analysis
	discriminant_analysis.LinearDiscriminantAnalysis(),
	discriminant_analysis.QuadraticDiscriminantAnalysis(),
	
	# XGBoost
	xgboost.XGBClassifier()
	   ]


vote_est = [('ada', ensemble.AdaBoostClassifier()), ('bc', ensemble.BaggingClassifier()),
	('etc', ensemble.ExtraTreesClassifier()), ('gbc', ensemble.GradientBoostingClassifier()),
	('rfc', ensemble.RandomForestClassifier()),
	
	('gpc', gaussian_process.GaussianProcessClassifier()),
	
	('lr', linear_model.LogisticRegressionCV()),
	
	# ('bnb', naive_bayes.BernoulliNB()), ('gnb', naive_bayes.GaussianNB()),
	
	('knn', neighbors.KNeighborsClassifier()),
	
	('svc', svm.SVC(probability=True)),
 
	('xgb', xgboost.XGBClassifier())]


# split method for cross validation
cv_split = model_selection.StratifiedShuffleSplit(n_splits=10, test_size=0.3, train_size=0.6, random_state=0)
	
# Create table to compare MLAs metrics
MLA_compare = pd.DataFrame(columns=['MLA Name', 'MLA Parameters', 'MLA Train Accuracy Mean', 'MLA Test Accuracy Mean',
                                    'MLA Test Accuracy 3*STD', 'MLA Time'])

# Create table to compare MLAs predictions
MLA_predict = pd.DataFrame(db_train['Survived'])

# Define data to use
# Simply Labeled
X = db_train.iloc[:,1:]
y = db_train.iloc[:,0]

# One hot encoding
# X = db_train_dummy.iloc[:,1:]

X.columns=['Pclass', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q',
       'Embarked_S', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs',
       'Title_Rare', 'Cat Family_1', 'Cat Family_2',
       'Cat Family_3', 'Cat Age_1', 'Cat Age_2',
       'Cat Age_3', 'Cat Age_4',
       'Cat Fare_1', 'Cat Fare_2',
       'Cat Fare_3', 'Cat Fare_4']

# y = db_train_dummy.iloc[:,0]


# Loop through classifiers and save performances
row_index = 0
for alg in MLA:
	# Set name and parameters
	MLA_name = alg.__class__.__name__
	MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
	MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
	
	# Evaluate the algorithm with cross validation
	cv_results = model_selection.cross_validate(alg, X, y, cv = cv_split)
	
	MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
	MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
	MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()
	MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std() * 3
	
	# Fit the model and get the predictions
	alg.fit(X, y)
	MLA_predict[MLA_name] = alg.predict(X)
	
	row_index += 1
	
#Hard Vote or majority rules
MLA_compare.loc[row_index, 'MLA Name'] = "Vote Hard"
vote_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')
vote_hard_cv = model_selection.cross_validate(vote_hard, X, y, cv  = cv_split)

MLA_compare.loc[row_index, 'MLA Time'] = vote_hard_cv['fit_time'].mean()
MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = vote_hard_cv['train_score'].mean()
MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = vote_hard_cv['test_score'].mean()
MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = vote_hard_cv['test_score'].std() * 3

vote_hard.fit(X, y)
MLA_predict["Vote Hard"] = vote_hard.predict(X)

row_index += 1

#Soft Vote or weighted probabilities
MLA_compare.loc[row_index, 'MLA Name'] = "Vote Soft"
vote_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')
vote_soft_cv = model_selection.cross_validate(vote_soft, X, y, cv  = cv_split)

MLA_compare.loc[row_index, 'MLA Time'] = vote_soft_cv['fit_time'].mean()
MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = vote_soft_cv['train_score'].mean()
MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = vote_soft_cv['test_score'].mean()
MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = vote_soft_cv['test_score'].std() * 3

vote_soft.fit(X, y)
MLA_predict["Vote Soft"] = vote_soft.predict(X)



MLA_compare.sort_values(by= ['MLA Test Accuracy Mean'], ascending=False, inplace=True)


# Barplot for algorithm comparison
plt.figure()
sns.barplot(x='MLA Test Accuracy Mean',  y = 'MLA Name', data = MLA_compare, color = 'b')
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')
plt.tight_layout()
plt.show()



clf = []
clf = [x for x in MLA if (x.__class__.__name__ == MLA_compare.iloc[0,0])][0]

if not clf:
	print('Fit Vote Hard or Soft and run predictions manually')
else:
	clf.fit(X, y)
	result = clf.predict(db_test)
	
	submit = pd.concat([pd.Series(test.index.values, name='PassengerID')+1,pd.Series(result, name='Survived')],axis=1)
	submit.to_csv("../Titanic/submit_1.csv", index=False)

'''
import torch
import torch.nn as nn
import torch.optim as optim
import PlotFunctions as PF
import sklearn.metrics as metrics
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import CNN

#CNN with pytorch
db_m = db_train_dummy.as_matrix()

#Create network, criterion and optimizer
net = CNN.Net()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


# Define procedure for training the network
def train(epoch):
	net.train()
	train_loss = 0
	acc_meas = [0, 0]
	b_id = 0
	total_sample, total_predictions = [], []
	
	# for each batch generated by DataLoader
	for batch_id, sample in enumerate(trainDataL):
		optimizer.zero_grad()  # Clear the gradients from our optimizer
		# inputs, targets = sample['X'], sample['Y']
		outputs = net(sample['X'].float())
		loss = criterion(outputs,
		                 sample['Y'].float().reshape(-1, 1))  # Calculate the loss between predictions and targets
		loss.backward()  # Get the gradient values
		optimizer.step()  # Update the weights of the optimizer with the gradients calculated
		
		train_loss += loss.item()  # add the loss of each batch to be able to calculate the overall loss
		predicted = (outputs.data >= 0.5) * 1
		
		total_sample += sample['Y'].tolist()
		total_predictions += predicted.reshape(-1).tolist()
		
		acc_meas[1] += sample['Y'].size(0)
		acc_meas[0] += metrics.accuracy_score(sample['Y'], predicted.reshape(-1), normalize=False)
		b_id += 1
	
	if epoch == 399:
		print(metrics.classification_report(total_sample, total_predictions))
	acc_tr[0].append(100. * acc_meas[0] / acc_meas[1])
	loss_tr[0].append(train_loss / (b_id))
	print('TRAIN Epoch %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
		epoch, train_loss / (b_id), 100. * acc_meas[0] / acc_meas[1], acc_meas[0], acc_meas[1]))


def test(epoch):
	net.eval()
	test_loss = 0
	acc_meas = [0, 0, 0]
	b_id = 0
	
	# for each batch generated by DataLoader
	for batch_id, sample in enumerate(testDataL):
		outputs = net(sample['X'].float())
		loss = criterion(outputs,
		                 sample['Y'].float().reshape(-1, 1))  # Calculate the loss between predictions and targets
		
		test_loss += loss.item()  # add the loss of each batch to be able to calculate the overall loss
		predicted = (outputs.data >= 0.5) * 1
		acc_meas[1] += sample['Y'].size(0)
		acc_meas[0] += metrics.accuracy_score(sample['Y'], predicted.reshape(-1), normalize=False)
		b_id += 1
	
	acc_tr[1].append(100. * acc_meas[0] / acc_meas[1])
	loss_tr[1].append(test_loss / (b_id))
	print('TEST  Epoch %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
		epoch, test_loss / (b_id), 100. * acc_meas[0] / acc_meas[1], acc_meas[0], acc_meas[1]))


#Define folds for cross validation
kf = model_selection.KFold(5)

for train_index, test_index in kf.split(db_m):
	X_train, X_test = db_m[train_index,1:], db_m[test_index,1:]
	y_train, y_test = db_m[train_index, 0], db_m[test_index, 0]

# Simple split without folding
# X_train, X_test, y_train, y_test = train_test_split(db_m[:,1:], db_m[:,0], test_size=0.2, random_state=42)

	# Transform our data into tensors to be read by our CNN and split it in shuffled batches
	batch_size = 16
	trainData = CNN.prepData(X_train, y_train)
	trainDataL = DataLoader(trainData, batch_size=batch_size)
	
	testData = CNN.prepData(X_test, y_test)
	testDataL = DataLoader(testData, batch_size=batch_size)
	
	# Initialize counters for accuracy and epoch
	acc_tr, loss_tr = [[], []], [[], []]
	start_epoch = 0

	nr_epochs = 400
	for epoch in range(start_epoch, start_epoch + nr_epochs):
		train(epoch)
		test(epoch)


t = np.arange(nr_epochs)

#Plot accuracy and loss for train and test sets for each epoch
fig, axes = plt.subplots(2,1)
ax1, ax2 = PF.two_y_axis(t,acc_tr[0],loss_tr[0], axes[0])
ax3, ax4 = PF.two_y_axis(t,acc_tr[1],loss_tr[1], axes[1])

#Set axes and colors
ax1.set_ylabel('Accuracy', color='b')
ax1.tick_params('y', colors='b')
ax1.set_title('Accuracy and Loss for Train Set')
ax2.set_ylabel('Loss', color='r')
ax2.tick_params('y', colors='r')

PF.setattrs(ax1.lines[0],'b')
PF.setattrs(ax2.lines[0],'r')

ax3.set_title('Accuracy and Loss for Test Set')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Accuracy', color='b')
ax3.tick_params('y', colors='b')
ax4.set_ylabel('Loss', color='r')
ax4.tick_params('y', colors='r')

PF.setattrs(ax3.lines[0],'b')
PF.setattrs(ax4.lines[0],'r')


'''
# Plot accuracy
fig, ax = plt.subplots()
plt.plot(t, acc_tr[0], 'b-')
plt.plot(t, acc_tr[1], 'r-.')
plt.title('Accuracy for Train and Test Sets')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy (%)')

labels = ['train', 'test']
plt.legend(labels, fancybox=True, shadow=True, labelspacing=0.0)

fig.tight_layout()
'''
plt.show()
