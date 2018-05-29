import pandas as pd
import re
import random


def c_age(db, pclass, title, family):
	# Filters the database according to the individual specifics
	temp_db = db[db['Pclass'] == pclass]
	if temp_db['Title'][temp_db['Title'] == title].count() != 0:
		temp_db = temp_db[temp_db['Title'] == title]
	if temp_db['Cat Family'][temp_db['Cat Family'] == family].count() != 0:
		temp_db = temp_db[temp_db['Cat Family'] == family]
	temp_db = temp_db[~temp_db['Cat Age'].isnull()]
	
	# Gets the Age categories and their frequency in filtered database
	cat = temp_db['Cat Age'].unique();	prob = []
	for i in cat:
		prob.append(temp_db[temp_db['Cat Age'] == i].shape[0])
	
	# Using the categories distribution, assign a random category
	random.seed(0)
	cat_age = random.choices(cat, prob)
	
	# Within the category defined previously assign a random Age
	age = random.uniform(cat_age[0]._repr_base()[0], cat_age[0]._repr_base()[1])
	if age > 1: age = int(age)
	return age, cat_age


# Function to get Title
def c_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	if title_search:
		return title_search.group(1)
	return ""


def clean_data(data):
	# Get Title
	data['Title'] = data['Name'].apply(c_title)
	data['Title'] = data['Title'].replace(['Mlle', 'Ms'], 'Miss')
	data['Title'] = data['Title'].replace('Mme', 'Mrs')
	data.loc[~data['Title'].isin(['Master', 'Miss', 'Mrs', 'Mr']), 'Title'] = "Rare"
	
	# Get Family/Group members count
	data['Family'] = pd.DataFrame({'Family': data['Parch'] + data['SibSp'] + 1,
	                               'Nr Ticket': [data['Ticket'][data['Ticket'] == data.loc[row, 'Ticket']].count() for
		                               row, item in data.iterrows()]}).apply(max, axis=1)
	
	# Create Family Categories
	data['Cat Family'] = pd.cut(data['Family'], [0, 1, 4, 12])
	data['IsAlone'] = 0;    data.loc[data['Family'] == 1, 'IsAlone'] = 1
	
	# Categorize Age in x bins - DEFINE BINS!
	data['Cat Age'] = pd.cut(data['Age'], 5)
	# Percentages of each category population before randomly assigning missing categories and ages
	# print(data['Cat Age'].groupby(data['Cat Age']).count() * 100 / data['Cat Age'].count())
	
	# Fill missing values for Cat Age and Age
	# Use Pclass, Title and Cat Family to filter results, get the distribution of Cat Age and assign cat according
	# to it. Assign a random number in the respective intervals to Age. Recategorize Age with integer numbers
	for row, item in data[data['Age'].isnull()].iterrows():
		data.loc[row, 'Age'], data.loc[row, 'Cat Age'] = c_age(data[~data['Age'].isnull()], item['Pclass'],
		                                                       item['Title'], item['Cat Family'])
	
	# Percentages of each category population after randomly assigning missing categories and ages
	# print(data['Cat Age'].groupby(data['Cat Age']).count() * 100 / data['Cat Age'].count())
	data['Cat Age'] = pd.cut(data['Age'].astype(int), 4)
	
	# Fare
	# We need to fill the only missing value of fare
	# For that we look at similar people (similar age category, pclass, Embarked and IsAlone) and find the mean (7,74)
	# Assign the value to the closest seen in this filtered dataframe (7.75000
	data['Fare'].fillna(7.75000, inplace=True)
	
	# Fare per person
	# There are tickets that have multiple people. In order to have a good comparison with people with single tickets we
	# need to look at the price per person
	tckt_cnt = data['Ticket'].groupby(data['Ticket']).count().to_dict()
	data['Fare per Person'] = data['Fare']/(data['Ticket'].replace(tckt_cnt).astype(int))
	
	# Categorize Fare per person
	data['Cat Fare'] = pd.cut(data['Fare per Person'], [-0.01,10,20,30,1000])
	
	# Fill in missing values on Embarked
	# Just two values missing so we perform a similar correction as for Fare
	# Both people have the same ticket so they embarked in the same place.
	# Filtering by Pclass, Sex, Cat Fare
	# Since the probability and the neighbours are distributed it will be assigned the most common boarding place (S)
	data['Embarked'].fillna('S', inplace=True)
	
	
	return data
