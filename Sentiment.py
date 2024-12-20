#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


#Load Dataset
df=pd.read_csv("D:\\Sem-5\\Project II\\twitter_dataset.csv", encoding='ISO-8859-1')


# In[4]:


#checking the number of rows and columns
df.shape


# In[5]:


# Understand the structure and content of the dataset
print("Dataset Info:")
print(df.info())


# In[6]:


#printing first five lines
print("\nFirst Few Rows of the Dataset:")
df.head()


# In[7]:


# Identify missing values
missing_values = df.isnull().sum()
print("\nMissing Values in Each Column:")
print(missing_values)


# In[8]:


#naming the columns and readinh the dataset
# Rename columns
columns_name = ['target', 'id', 'date', 'flag', 'user', 'text']


# In[9]:


df=pd.read_csv("D:\\Sem-5\\Project II\\twitter_dataset.csv",names=columns_name, encoding='ISO-8859-1')


# In[10]:


df.shape


# In[11]:


df.head()


# In[12]:


# Summarize key statistics for numerical columns
numerical_summary = df.describe()
print("\nSummary Statistics for Numerical Columns:\n")
print(numerical_summary)


# In[13]:


# Check for missing values
print(df.isnull().sum())


# In[14]:


import re


# In[15]:


from nltk.corpus import stopwords


# In[16]:


import nltk


# In[17]:


nltk.download('stopwords')


# In[18]:


print(stopwords.words('english'))


# In[19]:


#data processing


# In[20]:


#checking the distribution of target colum
df['target'].value_counts()


# In[21]:


#convert the target "4" to "1"
df.replace({'target': {4: 1}}, inplace=True)


# In[22]:


df['target'].value_counts()


# In[23]:


# 0..> Negative tweet
# 1..> positive tweet


# In[24]:


#stemming


# In[25]:


#it is the process of reducing aword to its Root word
# actor,actress,acting = act


# In[26]:


from nltk.stem.porter import PorterStemmer


# In[27]:


port_stem = PorterStemmer()


# In[28]:


def stemming(content):
    
    stemmed_content = re.sub('[^a-zA-Z]' , ' ' , content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    
    return stemmed_content


# In[29]:


df['stemmed_content'] = df['text'].apply(stemming)


# In[30]:


df.head()


# In[31]:


print(df['stemmed_content'])


# In[32]:


print(df['target'])


# In[33]:


#separating the data and label
x =df['stemmed_content'].values


# In[34]:


y= df['target'].values


# In[35]:


print(x)


# In[36]:


print(y)


# In[37]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


from sklearn.linear_model import LogisticRegression


# In[40]:


from sklearn.metrics import accuracy_score


# In[41]:


#splitting the data to training data and test data
x_train , x_test , y_train , y_test = train_test_split(x,y ,test_size = 0.2 , stratify = y , random_state = 2)


# In[42]:


print(x.shape , x_train.shape , x_test.shape)


# In[43]:


print(y.shape , y_train.shape , y_test.shape)


# In[44]:


print(x_train)


# In[45]:


print(x_test)


# In[46]:


#convert the textual data to numerical data
vectorizer  = TfidfVectorizer()


# In[47]:


x_train = vectorizer.fit_transform(x_train)


# In[48]:


x_test = vectorizer.transform(x_test)


# In[49]:


print(x_train)


# In[50]:


print(x_test)


# In[51]:


#training the model


# In[52]:


#logistic Regression
model = LogisticRegression(max_iter=1000) #number of time it itrater


# In[53]:


model.fit(x_train , y_train)


# In[54]:


#model Evalution


# In[55]:


#accuracy score on train data
x_train_prediction = model.predict(x_train)


# In[56]:


training_data_accuracy = accuracy_score(y_train , x_train_prediction)


# In[57]:


print(training_data_accuracy)


# In[58]:


x_test_prediction = model.predict(x_test)
test_accuracy = accuracy_score(y_test , x_test_prediction)


# In[59]:


print(test_accuracy)


# In[60]:


#saving the trained model
import pickle


# In[61]:


filename = 'trained_model.sav'
pickle.dump(model , open(filename , 'wb'))


# In[62]:


#using saved model for future prediction


# In[69]:


#loading the saved model
loaded_model = pickle.load(open("http:/localhost:8888/edit/trained_model.sav",'rb'))


# In[77]:


x_new =x_test[3]


# In[78]:


print(y_test[3])


# In[81]:


predication = model.predict(x_new)
print(predication)
if (predication[0] == 0):
    print("Negative Tweet")
    
else:
     print("Positive Tweet") 


# In[ ]:




