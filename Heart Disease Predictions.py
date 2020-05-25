#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Prediction
# 
# In this machine learning project, I have collected the dataset from Kaggle (https://www.kaggle.com/ronitf/heart-disease-uci) and I will be using Machine Learning to predict whether any person is suffering from heart disease

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# Here we will be experimenting with 3 algorithms 
# 1. KNeighborsClassifier
# 2. DecisionTreeClassifier
# 3. RandomForestClassifier
# 

# In[2]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[3]:


df = pd.read_csv('dataset.csv')


# In[4]:


df.info()


# In[6]:


df.describe()


# ## Feature Selection

# In[11]:


import seaborn as sns
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[14]:


df.hist()


# It's always a good practice to work with a dataset where the target classes are of approximately equal size. Thus, let's check for the same.

# In[16]:


sns.set_style('whitegrid')
sns.countplot(x='target',data=df,palette='RdBu_r')


# ### Data Processing
# 
# After exploring the dataset, I observed that I need to convert some categorical variables into dummy variables and scale all the values before training the Machine Learning models.
# First, I'll use the `get_dummies` method to create dummy columns for categorical variables.

# In[17]:


dataset = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])


# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])


# In[19]:


dataset.head()


# In[24]:


y = dataset['target']
X = dataset.drop(['target'], axis = 1)


# In[25]:


from sklearn.model_selection import cross_val_score
knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    score=cross_val_score(knn_classifier,X,y,cv=10)
    knn_scores.append(score.mean())


# In[26]:


plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
for i in range(1,21):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')


# In[36]:


knn_classifier = KNeighborsClassifier(n_neighbors = 12)
score=cross_val_score(knn_classifier,X,y,cv=10)


# In[38]:


score.mean()


# ## Random Forest Classifier

# In[27]:


from sklearn.ensemble import RandomForestClassifier


# In[30]:


randomforest_classifier= RandomForestClassifier(n_estimators=10)

score=cross_val_score(randomforest_classifier,X,y,cv=10)


# In[31]:


score.mean()


# In[ ]:





# In[ ]:




