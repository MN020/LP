#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import metrics


# In[3]:


df = pd.read_csv("diabetes.csv")
df


# In[4]:


df.shape


# In[5]:


df.head


# In[6]:


# checking for null values
df.isnull().any().value_counts()


# In[7]:


df.columns


# In[8]:


df_x = df.drop(columns='Outcome', axis=1)
df_y = df['Outcome']


# In[9]:


from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scaledX = scale.fit_transform(df_x)


# In[13]:


# split into train and test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(scaledX, df_y, test_size=0.2, random_state=42)



# In[14]:


# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)


# In[15]:


# Confusion matrix
cs = metrics.confusion_matrix(y_test,y_pred)
print("Confusion matrix: \n",cs)


# In[16]:


# Accuracy score
ac = metrics.accuracy_score(y_test, y_pred)
print("Accuracy score: ",ac)


# In[17]:


# Error rate (error_rate = 1- accuracy)
er = 1-ac
print("Error rate: ",er)


# In[19]:


# Precision
p = metrics.precision_score(y_test,y_pred)
print("Precision: ", p)


# In[20]:


# Recall
r = metrics.recall_score(y_test,y_pred)
print("Recall: ", r)


# In[22]:


# Classification report
cr = metrics.classification_report(y_test,y_pred)
print("Classification report: \n\n", cr)


# In[ ]:




