#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[7]:


df=pd.read_csv("Customer-Churn-Records.csv")


# In[8]:


df.head()


# In[10]:


df.shape


# In[11]:


df.describe()


# In[12]:


df.isnull()


# In[13]:


df.isnull().sum()


# In[14]:


df.info()


# In[15]:


df.dtypes


# In[16]:


df.columns


# In[18]:


df=df.drop(['RowNumber','Surname','CustomerId'],axis=1)


# In[19]:


df.head()


# In[21]:


def visualization(x, y, xlabel):
    plt.figure(figsize=(10,5))
    plt.hist([x, y], color=['red', 'green'], label = ['exit', 'not_exit'])
    plt.xlabel(xlabel,fontsize=20)
    plt.ylabel("No. of customers", fontsize=20)
    plt.legend()


# In[22]:


df_churn_exited = df[df['Exited']==1]['Tenure']
df_churn_not_exited = df[df['Exited']==0]['Tenure']


# In[23]:


visualization(df_churn_exited, df_churn_not_exited, "Tenure")


# In[24]:


df_churn_exited2 = df[df['Exited']==1]['Age']
df_churn_not_exited2 = df[df['Exited']==0]['Age']


# In[25]:


visualization(df_churn_exited2, df_churn_not_exited2, "Age")


# In[27]:


X = df[['CreditScore','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']]
states = pd.get_dummies(df['Geography'],drop_first = True)
gender = pd.get_dummies(df['Gender'],drop_first = True)


# In[28]:


df = pd.concat([df,gender,states], axis = 1)


# In[29]:


df.head()


# In[30]:


X = df[['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Male','Germany','Spain']]


# In[31]:


y = df['Exited']


# In[32]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.30)


# In[33]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[34]:


X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[35]:


X_train


# In[36]:


X_test


# In[41]:


pip install keras


# In[52]:


pip install tensorflow


# In[ ]:


import keras


# In[59]:


from keras.models import Sequential #To create sequential neural network
from keras.layers import Dense #To create hidden layers


# In[60]:


classifier = Sequential()


# In[61]:


#To add the layers
#Dense helps to contruct the neurons
#Input Dimension means we have 11 features
# Units is to create the hidden layers
#Uniform helps to distribute the weight uniformly
classifier.add(Dense(activation = "relu",input_dim = 11,units = 6,kernel_initializer = "uniform"))


# In[63]:


classifier.add(Dense(activation = "relu",units = 6,kernel_initializer = "uniform")) #Adding second hidden layers


# In[64]:


classifier.add(Dense(activation = "sigmoid",units = 1,kernel_initializer = "uniform")) #Final neuron will be having siigmoid function


# In[65]:


classifier.compile(optimizer="adam",loss = 'binary_crossentropy',metrics = ['accuracy'])
#To compile the Artificial Neural Network. Ussed Binary crossentropy as we just have onlytwo output


# In[66]:


classifier.summary() #3 layers created. 6 neurons in 1st,6neurons in 2nd layer and 1 neuron in last


# In[67]:


classifier.fit(X_train,y_train,batch_size=10,epochs=50) #Fitting the ANN to training data
set


# In[68]:


y_pred =classifier.predict(X_test)
y_pred = (y_pred > 0.5) #Predicting the result


# In[69]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[70]:


cm = confusion_matrix(y_test,y_pred)


# In[71]:


cm


# In[72]:


accuracy = accuracy_score(y_test,y_pred)


# In[73]:


accuracy


# In[74]:


plt.figure(figsize = (10,7))
sns.heatmap(cm,annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[75]:


print(classification_report(y_test,y_pred))


# In[ ]:




