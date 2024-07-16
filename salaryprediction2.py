#!/usr/bin/env python
# coding: utf-8

# In[100]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[101]:


df=pd.read_csv('C:\\Users\\rajan\\OneDrive\\Desktop\\Group10\\SalaryData.csv')
df.head()


# In[102]:


df.plot(x='TotalWorkingYears',y='MonthlyIncome',style='o')
plt.title('Working Years vs Monthly Salary')
plt.xlabel('TotalWorkingYears')
plt.ylabel('Salary')
plt.show()


# In[103]:


df.plot(x='Age',y='MonthlyIncome',style='o')
plt.title('Age vs Monthly Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()


# In[104]:


df['Department'].unique()


# In[105]:


df['JobRole'].unique()


# In[106]:


df.isnull().count()


# In[107]:


df['Department']=df['Department'].map({'Sales':1,'Research & Development':2,'Human Resources':3})


# In[108]:


df['Gender']=df['Gender'].map({'Male':1,'Female':2})


# In[109]:


df['JobRole']=df['JobRole'].map({'Laboratory Technician':1,'Research Scientist':2,'Manufacturing Director':3,'Healthcare Representative':4,'Manager':5,'Sales Representative':6,'Research Director':7,'Sales Executive':8,'Human Resources':9})


# In[110]:


df.head()


# In[111]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[112]:


from sklearn.model_selection import train_test_split


# In[113]:


X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.19, random_state=42)


# In[114]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


# In[115]:


y_pred=regressor.predict(X_test)


# In[116]:


df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})


# In[117]:


df


# In[118]:


from sklearn.metrics import r2_score


# In[119]:


r2_score(y_test, y_pred)


# In[120]:


import joblib


# In[121]:


joblib.dump(regressor,'linear_regression_model.pkl')


# In[ ]:




