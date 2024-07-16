#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import joblib
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[26]:


df=pd.read_csv("SalaryData.csv")


# In[27]:


df.head()


# In[28]:


df.shape


# In[67]:


df = df[~((df['Department'] == 'Human Resources') & (df['JobRole'].str.contains('Human Resource')))]
df = df[df['Department'] != 'Human Resources']


# In[68]:


df.Department.value_counts()


# In[30]:





# In[69]:


df.value_counts()


# In[ ]:





# In[70]:


df['Age'].hist()


# In[71]:


df['Gender'].hist()


# In[72]:


df['Department'].hist()


# In[73]:


sns.countplot(x='Gender', data=df)


# In[74]:


df['Department'].value_counts().plot(kind='bar')


# In[75]:


plt.scatter(df['Age'], df['MonthlyIncome'])


# In[76]:


num_col=[x for x in df.columns if df[x].dtype!='object']
cat_col=[x for x in df.columns if df[x].dtype=='object']


# In[77]:


num_col


# In[78]:


cat_col


# In[79]:


def get_unique(columns):
    for i in columns:
        print(f'{i}=====>{df[i].unique()}\n')


# In[80]:


get_unique(cat_col)


# In[81]:


df_new=pd.get_dummies(df,columns=cat_col,drop_first=False)


# In[82]:


df_new.info()


# In[83]:


cols=[x for x in df_new.columns if x !="MonthlyIncome"]


# In[84]:


cols


# In[85]:


import sklearn
from sklearn.preprocessing import StandardScaler


# In[86]:


sc=StandardScaler()
sc.fit_transform(df_new)


# In[87]:


from sklearn.model_selection import train_test_split


# In[88]:


df_train,df_test=train_test_split(df_new,test_size=0.3,random_state=42)


# In[89]:


df_train.shape


# In[90]:


df_test.shape


# In[91]:


X_train=df_train[cols]
X_test=df_test[cols]


# In[92]:


y_train=df_train['MonthlyIncome']
y_test=df_test['MonthlyIncome']


# In[93]:


df_train.head()


# In[94]:


from sklearn.linear_model import LinearRegression


# In[95]:


lr=LinearRegression()


# In[96]:


lr.fit(X_train,y_train)


# In[97]:


y_pred=lr.predict(X_train)


# In[98]:


lr.score(X_train,y_train)


# In[99]:


lr.score(X_test,y_test)


# In[100]:


score = lr.score(X_test, y_test)
print(f"Model R^2 Score: {score}")


# In[101]:


joblib.dump(lr, 'linear_regression_model.pkl')


# In[ ]:





# In[ ]:




