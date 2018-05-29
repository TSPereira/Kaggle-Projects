import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('databases/train.csv')
test = pd.read_csv('databases/test.csv')

data = [train, test]

for d in data:
	d.drop('Id',axis=1, inplace=True)

# Analyse target
plt.subplot(1,2,1)
sns.distplot(train.SalePrice)

plt.subplot(1,2,2)
sns.distplot(np.log(train.SalePrice))
plt.show()


df_num = train.select_dtypes(include=['float64','int64'])
