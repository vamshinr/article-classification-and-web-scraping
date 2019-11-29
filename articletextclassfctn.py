#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import seaborn as sns
sns.set_style("whitegrid")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn import svm
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit


# In[ ]:


df=pd.read_csv('News_dataset.csv',sep=';')


# In[ ]:


df.head()


# In[29]:


df.groupby('Category').id.count().plot.bar(ylim=0)
plt.show()


# In[ ]:


df['id']=1


# In[28]:


df.head()


# In[36]:


df.loc[0]['Content']


# In[33]:


df['cp_1']=df['Content'].str.replace("\r"," ")
df['cp_1']=df['cp_1'].str.replace("\n"," ")
df['cp_1']=df['cp_1'].str.replace("    "," ")
df['cp_1']=df['cp_1'].str.replace('"','')


# In[34]:


df.loc[0]['Content']


# In[40]:


df['cp_2']=df['cp_1'].str.lower()


# In[41]:


psigns=['?','!',';',':','.',',']
df['cp_3']=df['cp_2']
for pun in psigns:
    df['cp_3']=df['cp_3'].str.replace(pun,'')


# In[42]:


df['cp_4']=df['cp_3'].str.replace("'s","")


# In[43]:


stop=list(stopwords.words("english"))


# In[44]:


type(stop)


# In[47]:


nltk.download('punkt')
nltk.download('wordnet')


# In[48]:


wl=WordNetLemmatizer()


# In[49]:


l=len(df)
lem_text_list=[]
for i in range(l):
    lem_list=[]
    text=df.loc[i]['cp_4']
    text_words=text.split()
    for word in text_words:
        if word not in stop:
            words=wl.lemmatize(word)
            lem_list.append(words)
    lem_text=' '.join(lem_list)
    lem_text_list.append(lem_text)


# In[50]:


df['cp_5']=df['cp_4']
df['cp_5']=lem_text_list


# In[51]:


colnames=["File_Name", "Category", "Complete_Filename", "Content", "cp_4"]
df=df[colnames]


# In[52]:


df=df.rename(columns={'cp_4':'Cleancontent'})


# In[53]:


df.loc[0]['Content']


# In[54]:


df.loc[0]['Cleancontent']


# In[55]:


category_codes = {
    'business': 0,
    'entertainment': 1,
    'politics': 2,
    'sport': 3,
    'tech': 4
}


# In[56]:


df['Category_Code'] = df['Category']
df = df.replace({'Category_Code':category_codes})


# In[57]:


X_train, X_test, y_train, y_test = train_test_split(df['Cleancontent'],df['Category_Code'],test_size=0.15,random_state=8)


# In[58]:


ngram_range = (1,2)
min_df = 10
max_df = 1.
max_features = 300
tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)
                        
X_newtrain = tfidf.fit_transform(X_train).toarray()
y_newtrain = y_train
print(X_newtrain.shape)

X_newtest = tfidf.transform(X_test).toarray()
y_newtest=y_test
print(X_newtest.shape)


# In[59]:


SVM =svm.SVC(C=1.0,kernel='linear',degree=3,gamma='auto',probability=True)
SVM.fit(X_newtrain,y_newtrain)
svm_pred=SVM.predict(X_newtest)


# In[60]:


accuracy_score(svm_pred,y_newtest)*100


# In[61]:


category_names = {
    0: 'business',
    1: 'entertainment',
    2: 'politics',
    3: 'sport',
    4: 'tech'
}
# Indexes of the test set
index_X_test = X_test.index

# We get them from the original df
df_test = df.loc[index_X_test]

# Add the predictions
df_test['Prediction'] = svm_pred

# Clean columns
df_test = df_test[['Content', 'Category', 'Category_Code', 'Prediction']]

# Decode
df_test['Category_Predicted'] = df_test['Prediction']
df_test = df_test.replace({'Category_Predicted':category_names})

# Clean columns again
df_test = df_test[['Content', 'Category', 'Category_Predicted']]


# In[62]:


df_test.head()


# In[63]:


condition = (df_test['Category'] != df_test['Category_Predicted'])

df_misclassified = df_test[condition]

df_misclassified.head(3)


# In[64]:


text="""

The center-right party Ciudadanos closed a deal on Wednesday with the support of the conservative Popular Party (PP) to take control of the speaker’s committee in the Andalusian parliament, paving the way for the regional PP leader, Juan Manuel Moreno, to stand as the candidate for premier of the southern Spanish region. The move would see the Socialist Party (PSOE) lose power in the Junta, as the regional government is known, for the first time in 36 years.

Talks in Andalusia have been ongoing since regional polls were held on December 2. The PSOE, led by incumbent premier Susana Díaz, had been expected to win the early elections, but in a shock result the party took the most seats in parliament, 33, but fell well short of a majority of 55. It was their worst result in the region since Spain returned to democracy. The PP came in second, with 26 seats, while Ciudadanos were third with 21. The major surprise was the strong performance of far-right group Vox, which won more than 391,000 votes (10.9%), giving it 12 deputies. The anti-immigration group is the first of its kind to win seats in a Spanish parliament since the end of the Francisco Franco dictatorship. It now holds the key to power in Andalusia, given that its votes, added to those of the PP and Ciudadanos, constitute an absolute majority.

The move would see the Socialist Party lose power in the region for the first time in 36 years

On Thursday, Marta Bosquet of Ciudadanos was voted in as the new speaker of the Andalusian parliament thanks to 59 votes from her party, the PP and Vox. The other candidate, Inmaculada Nieto of Adelante Andalucía, secured 50 votes – from her own party and 33 from the PSOE.

The speaker’s role in the parliament is key for the calling of an investiture vote and for the selection of the candidate for premier.

Officially, the talks as to the make up of a future government have yet to start, but in reality they are well advanced, according to sources from both the PP and Ciudadanos. The leader of the Andalusian PP is banking on being voted into power around January 16 and wants the majority of his Cabinet to be decided “five days before the investiture vote.”

The speaker’s role in the parliament is key for the calling of an investiture vote and for the selection of the candidate for premier

The PP, which was ousted from power by the PSOE in the national government in June, is keen to take the reins of power in Andalusia as soon as possible. The difficulties that Ciudadanos has faced to justify the necessary inclusion of Vox in the talks, has slowed down progress. Rather than align itself with the far right party, the group – which began life in Catalonia in response to the independence drive, but soon launched onto the national stage – had sought a deal with Adelante Andalucía.

Wednesday was a day of intense talks among the parties in a bid to find a solution that would keep everyone happy. But at 9pm last night, Adelante Andalucía announced that it would not be part of “any deal” and that would instead vote for its own candidates to the speaker’s committee in order to “face up to the right wing and the extreme right.”

The PSOE, meanwhile, argues that having won the elections with a seven-seat lead over the PP gives it the legitimacy to aspire to the control of the regional government and the parliament, and to maintain its positions on the speaker’s committee.


"""


# In[65]:


punctuation_signs = list("?:!.,;")
stop_words = list(stopwords.words('english'))

def create_features_from_text(text):
    
    # Dataframe creation
    lemmatized_text_list = []
    df = pd.DataFrame(columns=['Content'])
    df.loc[0] = text
    df['Content_Parsed_1'] = df['Content'].str.replace("\r", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("\n", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("    ", " ")
    df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace('"', '')
    df['Content_Parsed_2'] = df['Content_Parsed_1'].str.lower()
    df['Content_Parsed_3'] = df['Content_Parsed_2']
    for punct_sign in punctuation_signs:
        df['Content_Parsed_3'] = df['Content_Parsed_3'].str.replace(punct_sign, '')
    df['Content_Parsed_4'] = df['Content_Parsed_3'].str.replace("'s", "")
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_list = []
    text = df.loc[0]['Content_Parsed_4']
    text_words = text.split(" ")
    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
    lemmatized_text = " ".join(lemmatized_list)    
    lemmatized_text_list.append(lemmatized_text)
    df['Content_Parsed_5'] = lemmatized_text_list
    df['Content_Parsed_6'] = df['Content_Parsed_5']
    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        df['Content_Parsed_6'] = df['Content_Parsed_6'].str.replace(regex_stopword, '')
    df = df['Content_Parsed_6']
    df = df.rename(columns={'Content_Parsed_6': 'Content_Parsed'})
    
    # TF-IDF
    features = tfidf.transform(df).toarray()
    
    return features


# In[66]:


def get_category_name(category_id):
    for category, id_ in category_codes.items():    
        if id_ == category_id:
            return category


# In[ ]:





# In[67]:


def predict_from_text(text):
    
    # Predict using the input model
    prediction_svc = SVM.predict(create_features_from_text(text))[0]
    prediction_svc_proba = SVM.predict_proba(create_features_from_text(text))[0]
    # Return result
    category_svc = get_category_name(prediction_svc)
    print(category_svc)
    print(prediction_svc_proba.max()*100)
    print(type(prediction_svc_proba))
    print(type(prediction_svc))


# In[68]:


predict_from_text(text)


# In[ ]:


pred=SVM.predict(create_features_from_text(text))
pred


# In[ ]:


SVM.predict_proba(create_features_from_text(text))[0]


# In[ ]:


prediction_svc


# In[ ]:




