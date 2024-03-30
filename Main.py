#!/usr/bin/env python
# coding: utf-8

# In[52]:



# https://youtu.be/dhoKFqhVJu0?si=BU7Dznb1EQZspgFy

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[53]:


df = pd.read_csv('income.csv')


# In[54]:


df


# In[55]:


df.education.value_counts()


# In[56]:


df.workclass.value_counts()


# In[57]:


df.occupation.value_counts()


# In[58]:


df.gender.value_counts()


# In[59]:


pd.get_dummies(df.occupation).add_prefix('occupation_')   #One hot encoding occupation using pandas


# In[60]:


df=pd.concat([df.drop('occupation', axis=1),pd.get_dummies(df.occupation).add_prefix('occupation_')],axis=1)   #Adding it to main df


# In[61]:


df


# In[62]:


df=pd.concat([df.drop('workclass', axis=1),pd.get_dummies(df.workclass).add_prefix('workclass_')],axis=1)


# In[63]:


df


# In[64]:


df=df.drop('education',axis=1)


# In[65]:


df


# In[66]:


df=pd.concat([df.drop('marital-status', axis=1),pd.get_dummies(df['marital-status']).add_prefix('marital-status_')],axis=1)


# In[67]:


df


# In[68]:


df=pd.concat([df.drop('relationship', axis=1),pd.get_dummies(df.relationship).add_prefix('relationship_')],axis=1)


# In[69]:


df


# In[70]:


df=pd.concat([df.drop('race', axis=1),pd.get_dummies(df.race).add_prefix('race_')],axis=1)


# In[71]:


df


# In[72]:


df=pd.concat([df.drop('native-country', axis=1),pd.get_dummies(df['native-country']).add_prefix('native-country_')],axis=1)


# In[73]:


df


# In[74]:


#Binary encoding gender and income
df['gender']=df['gender'].apply(lambda x:1 if x=='Male' else 0)
df['income']=df['income'].apply(lambda x:1 if x=='>50K' else 0)


# In[75]:


df


# In[76]:


df.columns.values


# In[77]:


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=False,cmap='coolwarm')


# In[78]:


df.corr()


# In[80]:


correlations = df.corr()['income'].abs()
sorted_correlations = correlations.sort_values()
num_cols_to_drop = int(0.8 * len(df.columns))
cols_to_drop = sorted_correlations.iloc[:num_cols_to_drop].index
df_dropped = df.drop(cols_to_drop, axis=1)


# In[84]:


df_dropped


# In[86]:


plt.figure(figsize=(15,10))
sns.heatmap(df_dropped.corr(),annot=True,cmap='coolwarm')


# In[100]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = df.drop('fnlwgt',axis=1)

train_df,test_df = train_test_split(df,test_size = 0.2)


# In[101]:


train_df


# In[102]:


test_df


# In[103]:


train_X = train_df.drop('income',axis=1)
train_y = train_df['income']

test_X = test_df.drop('income',axis=1)
test_y = test_df['income']


# In[104]:


forest = RandomForestClassifier()

forest.fit(train_X,train_y)


# In[105]:


forest.score(test_X, test_y)


# In[106]:


forest.feature_importances_


# In[107]:


forest.feature_names_in_


# In[108]:


importances = dict(zip(forest.feature_names_in_,forest.feature_importances_))
importances = {k:v for k, v in sorted(importances.items(),key=lambda x: x[1], reverse=True)}


# In[109]:


importances


# In[114]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 250],
    'max_depth': [5, 10, 30, None],
    'min_samples_split': [2, 4],
    'max_features': ['sqrt', 'log2']  # Corrected format
}


grid_search = GridSearchCV(estimator=RandomForestClassifier(),
                          param_grid=param_grid,verbose=10)


# In[115]:


grid_search.fit(train_X,train_y)


# In[116]:


grid_search.best_estimator_


# In[118]:


forest = grid_search.best_estimator_


# In[119]:


forest.score(test_X,test_y)


# In[120]:


importances = dict(zip(forest.feature_names_in_,forest.feature_importances_))
importances = {k:v for k, v in sorted(importances.items(),key=lambda x: x[1], reverse=True)}


# In[121]:


importances


# In[ ]:




