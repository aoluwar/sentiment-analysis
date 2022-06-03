# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 11:12:24 2022

@author: USER
"""
#libraries to be used
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import rcParams
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')


import re
import nltk
import string
import nlp_utils
import collections 
import contractions
import nlp_utils as nu
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from nltk.tokenize import word_tokenize,sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn .linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#READING FILE 

with open('C:/Users/USER/sentiment analysis/TextAnalytics.txt', 'r') as f:
    text = f.read()
    
#text normalization

#splitting at( \n)
text=text.split('\n')
#separating at the line using '\n'

#splitting at (\t)
corpus = [text]
for sent in text:
    corpus.append(sent.split('\t'))
#splitting String by tab(\t)

letters_only = re.sub(r'[^a-zA-Z]',
                      " ",
                      str(corpus))
#taking only letters

#TOKENIZATION
letters_only=letters_only.lower()

token=nltk.sent_tokenize(letters_only)
token

#Dealing with Alphanumeric characters
def num_dec_al(word):
    if word.isnumeric():
        return 'xxxxxx'
    elif word.isdecimal():
        return 'xxx...'
    elif word.isalpha():
        return word
    else:
        return 'xxxaaa'
    
def clean_nda(token):
    tokens = nlp_utils.w_tokenization(token)
    map_list = list(map(num_doc_al,tokens))
    return " ".join(map_list)

corpus_nda = list(map(clean_nda,token))

corpus_nda
###alphanumeric datas and decimal characters have been replaced with charaters
    
#contraction expansion
conn = contractions.CONTRACTION_MAP
def contraction_remove(corpus_nda):
    for key.value in conn.items():
        corpus_nda = re.sub(r"{}".format(key),'{}'(value),corpus_nda)
        
    return corpus_nda

special = string.punctuation
def w_tokenization(corpus_nda):
    #convert into lower case
    corpus_nda = corpus_nda.lower()
    #contraction 
    corpus_nda = contraction_remove(corpus_nda)
    #token
    tokens = nltk.word_tokenize(corpus_nda)
       
corpus_nda


data =[corpus_nda]
for sent in text:
    data.append(sent.split('\t'))
    
data.append(sent.split('\n'))

data

df = pd.DataFrame(data)
df

df.drop([1,2,3,4,5,6],axis=1, inplace=True)

df

df = df.rename(columns={0: 'text'})

df

df= df.rename(columns={0: 'Text'})
df

df.drop(df.index[:1], inplace=True)

df.drop(df.index[1000:], inplace=True)
df
#removing additional charaters present in the dataframe.
df.replace('\d', '',regex=True,  inplace=True)
df.replace(',', '',regex=True,  inplace=True)
df.replace('br', '',regex=True,  inplace=True)
df.replace('"', '',regex=True,  inplace=True)
df.replace('""', '',regex=True,  inplace=True)
df.replace('?', '',regex=True,  inplace=True)
df.replace(" . ", '',regex=True,  inplace=True)
df.replace("*", '',regex=True,  inplace=True)
df.replace("***", ' ',regex=True,  inplace=True)
df.replace("< />", '',regex=True,  inplace=True)

df['Text'] = df['Text'].str.strip('[')
df['Text'] = df['Text'].str.strip(']')
df['Text'] = df['Text'].str.strip('(')
df['Text'] = df['Text'].str.strip(')')

df

#lemmatization
df['Text'] = df['Text'].apply(nu.lemmatization_sentence)
Text=df['Text']

token=Text.as_matrix(columns=None)
#as_matrix() function is used to convert the the given series or dataframe object to numpy-array representation

token=nltk.sent_tokenize(str(token))
##Sentence tokenization

data = np.array(token)
##Saving  token in form of array

stop = stppwords.words('english')
##saving stopwords in stop

#removing stopwords from the dataframe 
text = data
text_tokens = words_tokenize(str(text))
tokens_without_sw = [word for word in text_tokens if not word is stop]
print(tokens_without_sw)
#print word without stopword

#visualization
stopwords = set(stopwords.words('english'))
##Removing stopwords for wordcloud visualization

wordcloud = wordCloud(stopwords=stop, background_color="black", max_words=1000).generate(str(tokens_without_sw))
#wordcloud is a technique to show which words are the most frequent among the given text

rcParams['figure.figure'] = 10,20
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

#counting the number of times a word is repeated through out the data
tokens_without_sw(str(tokens_without_sw))
filtered_words = [word for word in tokens_without_sw.split() if word not in stopwords]
counted_words =  collections.counter(filtered_words)

words = []
counts = []
for letter, count in counted_words.most_common(10):
    words.append(letter)
    counts.append(count)
#Removing stopwords as creating two lists to display the words and their counts

counted_words.most_common(100)

#Sentiment Analysis
#

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sentence = tokens_without_sw
tokenized_sentence = nltk.word_tokenize(sentence)

sid = SentimentIntensityAnalyzer()
pos_word_list=[]
neg_word_list=[]
neu_word_list=[]

for word in tokenized_sentence:
    if (sid.polarity_scores(word)['comound']) >= 0.1:
        pos_word_list.append(word)
    elif (sid.polarity_scores(word)['comound']) <= 0.1:
        neg_word_list.append(word)
    else:
        neu_word_list.append(word)
        
        #print ('posiive:',pos_word_list)
        #print ('negative:',neg_word_list)
        #print ('neutral:',neu_word_list)
        #score = sid.polarity_scores(sentence)
        #print ('\nScores:', score)
        
        
#testing for top 200 positive words.
print(list(iter(pos_word_list[:200]))) #top 200 words in the dataset

#testing for top 200 negative words.
print(list(iter(neg_word_list[:200]))) #top 200 words in the dataset

#Vader sentiment analysis of sentence
sid = SentimentIntensityAnalyzer
for sentence in Text:
    print(sentence)
    
    ss = sid.polarity_scores(sentence)
    for k in ss:
        print('(0):, (1),' .format(k, ss[k]), end='')
    print()
    
#converting polarity scores and the sentences into a dataframe
analyzer = SentimentIntensityAnalyzer
df['rating'] = Text.apply(analyzer.polarity_scores)
df= pd.concat([df.drop(['rating'], axis=1), df['rating'].apply(pd.series)], axis=1)
#creating a dataframe
df.head()

# Arranging the dataset in descending order based on (Compound score) to find the most important sentence from the given data
imp_sent=df.sort_values(by='compound', ascending=False)
## arranging the compound column is descending order  to find the best sentence
imp_sent


print(df['Text'].iloc[410]) #sentence with index 410 has the highest compound score 

#finding top pos, neg, neu sentence in the data
pos_sent=df.sort_value(by='pos', ascending=False)
pos_sent

print(df['Text'].iloc[160]) # sentence with index 160 has the highest positive score sentence

neg_sent=df.sort_value(by='pos', ascending=False)
neg_sent

print(df['Text'].iloc[413]) # sentence with index 413 has the highest negative score sentence
sentences=df

#giving threshold values to classify if a given sentence is positive, negative or nuetral in nature
#Assigning score categories and logic
i = 0

predited_value = [ ]

while(i<len(sentences)):
    if ((sentences.iloc[i]['compound'] >=0.5)):
        predicted_value.append('positive')
        i =i+1
    elif ((sentences.iloc[i]['compound'] > 0) & (sentences.iloc[i]['compound'] < 0.5)):
        predicted_value.append('neutral')
        i =i+1
    elif ((sentences.iloc[i]['compound'] <= 0)):
        predicted_value.append('negative')
        i =i+1
        
##threshold  value will be categorize if a given sentence is positive, negative or neutral in nature

predicted_value

#Adding the target or the sentiment to our dataframe
df['Target'] = predicted_value

df.head()
 
#Dropping the neg, neu,pos and compound columns
df.drop(['neg','neu','pos','compound'], axis=1, inplace=True) 
##Dropping the neg,neu,pos, and compound columns
df

df['Target'].value_counts()
###numbers of po, neg,neu, 

cat_cols=['Target']
le=LabelEncoder()
for i in cat_cols:
    df[i]=le.fit_transform(df[i])
    df.dtypes
##label encoding the target column.
df

#vectorizing training data.
tfidf = TfidfVectorizer()
x = tfidf.fit_transform(df['Text'])
y = df['Target']
##Applying TF-idf vectorizer on the Text column.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
###splitting the dataset

#Models
#logistic Regression
log_reg = LogisticRegression

#predict on train
train_preds = log_reg.predict(x_train)
#accuracy on Train
print("Model acccuracy on train is:", accuracy_score(y_train, train_preds))

#predict on test
  test_preds = log_reg.predict(x_test)
#accuracy on Test
print("Model acccuracy on test is:", accuracy_score(y_test, test_preds))


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_auc_score, ruc_curve, recall_score

#DECISION TREE CLASSIFIERS
train_preds2 = DT.predict(x_train)
#accuracy on train
print("Model accuracy on train is:", accuracy_score(y_train,train_preds2))

test_preds2 = DT.predict(x_test)
#accuracy on test
print("Model accuracy on train is:", accuracy_score(y_test,test_preds2))
print('-'*50)

#confusion matrix
print("confusion_matrix train is:", confusion_matrix(y_train, train_preds2))
print("confusion_matrix test is:", confusion_matrix(y_test, test_preds2))
print('Wrong prediction out in total')
print('-'*50)

#wrong prediction mode:
print((y_test !=test_preds2).sum(),'/',((y_test == test_preds2).sum()+(y_test != test_preds2).sum()))
print('-'*50)
#kappa Score
print('kappaScore is:', metrics.cohens_kappa_score(y_test,test_preds2))


#RandomForerstClassifier
#fit the model on train data
RF=RadomForestClassifier().fit(x_train,y_train)
#train
train_preds3 = RF.predict(x_train)
#accuracy on train 
print("model accuracy on train is :", accuracy_score(y_train,train_preds3))

test_preds3 = RF.predict(x_test)
#accuracy on train 
print("model accuracy in test is :", accuracy_score(y_test,test_preds3))
print('-'*50)

#confusion matrix
print("confusion_matrix train is:", confusion_matrix(y_train, train_preds3))
print("confusion_matrix test is:", confusion_matrix(y_test, test_preds3))
print('Wrong prediction out in total')
print('-'*50)

#wrong prediction mode:
print((y_test !=test_preds3).sum(),'/',((y_test == test_preds3).sum()+(y_test != test_preds3).sum()))
print('-'*50)

#wrong prediction mode:
print((y_test !=test_preds3).sum(),'/',((y_test == test_preds3).sum()+(y_test != test_preds3).sum()))
print('-'*50)

#KNN
#KNN fit on train data
KNN = KNeighborsClassifier().fit(x_train,y_train)
#train
train_preds4 = KNN.predict(x_train)
#accuracy on train 
print("model accuracy on train is :", accuracy_score(y_train,train_preds4))

test_preds4 = KNN.predict(x_test)
#accuracy on test 
print("model accuracy in test is :", accuracy_score(y_test,test_preds4))
print('-'*50)

#confusion matrix
print("confusion_matrix train is:", confusion_matrix(y_train, train_preds4))
print("confusion_matrix test is:", confusion_matrix(y_test, test_preds4))
print('Wrong prediction out in total')
print('-'*50)

#wrong prediction mode:
print((y_test !=test_preds4).sum(),'/',((y_test == test_preds4).sum()+(y_test != test_preds4).sum()))
print('-'*50)

#wrong prediction mode:
print((y_test !=test_preds4).sum(),'/',((y_test == test_preds4).sum()+(y_test != test_preds4).sum()))
print('-'*50)

#SVM
#fit the model on the train data
SVM = SVC(kernel='linear')
SVM.fit(x_train,y_train)

train_preds4 = KNN.predict(x_train)
#accuracy on train 
print("model accuracy on train is :", accuracy_score(y_train,train_preds5))

test_preds5 = SVM.predict(x_test)
#accuracy on test 
print("model accuracy in test is :", accuracy_score(y_test,test_preds5))
print('-'*50)

#confusion matrix
print("confusion_matrix train is:", confusion_matrix(y_train, train_preds5))
print("confusion_matrix test is:", confusion_matrix(y_test, test_preds5))
print('Wrong prediction out in total')
print('-'*50)

#wrong prediction mode:
print((y_test !=test_preds5).sum(),'/',((y_test == test_preds5).sum()+(y_test != test_preds5).sum()))
print('-'*50)

#wrong prediction mode:
print((y_test !=test_preds5).sum(),'/',((y_test == test_preds5).sum()+(y_test != test_preds5).sum()))
print('-'*50)



      
        


        
