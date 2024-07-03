from google.colab import userdata
from os import environ

environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')

!pip install -q kaggle 
!kaggle competitions download -c house-prices-advanced-regression-techniques
! unzip /content/house-prices-advanced-regression-techniques.zip

import pandas as pd

data = pd.read_csv('train.csv')
data
data.columns.to_list()

from sklearn.preprocessing import LabelEncoder

for col in data.columns:

  if pd.api.types.is_numeric_dtype(data[col].dtype):
    if data[col].isnull().any():
      data[col].fillna(data[col].median(), inplace = True)

  else:
    data[col] = LabelEncoder().fit_transform(data[col])

data

# Finding correlation between variables and visualizing it using seaborn

import matplotlib.pyplot as plt
import seaborn as sns

features = data.corr()[['SalePrice']].sort_values(by = ['SalePrice'], ascending = False).head(30)
plt.figure(figsize = [5,10])
sns.heatmap(features, cmap = 'rainbow', annot = True, vmin = -1)

# Setting up machine learning

y = data['SalePrice']
X = data.drop('SalePrice', axis = 'columns')
X = data.drop('Id' , axis = 'columns')

y, X

# Train test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 10)
y_test

from sklearn.linear_model import LinearRegression

traindata = LinearRegression()
traindata.fit(X_train, y_train)

traindata.score(X_train, y_train)
