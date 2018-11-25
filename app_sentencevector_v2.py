import  os
import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import numpy as np
import cgi
from datacleaner import *

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile

debug = 1
os.listdir(os.getcwd())

EMBEDDING_DIM=200

input_file = 'tweet_remove_blank.csv'


# In[2]:


debug = 1
os.listdir(os.getcwd())

EMBEDDING_DIM=200

input_file = 'tweet_remove_blank.csv'

count = 0

listTokens = []
maxvocab = 0
diclabel = {}
listlabels = []
with open(input_file,  encoding='utf-8') as f:
    for i,line in enumerate (f):

        count = count + 1
        
        if (count == 1):
            continue

        
        id, text, label  = tweettoidtextlabel(line)
        text = text.strip()
        if label in diclabel:
            val = diclabel[label]
            val = val +1 
            diclabel[label] =  val
        else:
            diclabel[label] = 0
        
        #print( "original tweet : [ " , count ," ] " ,  text )
        htmltext = removehtmlentities(text)
        urltext  = removeurl(htmltext)
        attext = removeat(urltext)
        triggertext =  removetriggerword(attext)
        expandabbrevation(triggertext)
        finaltoken = removestopword(tokenizeString(triggertext))
        curtotal = len(finaltoken)
        if ( curtotal > maxvocab):
            maxvocab = curtotal
        listTokens.append(finaltoken )
        listlabels.append(label)
        #print( "tweet set : [ " , count ," ] " ,  finaltoken )
        #print('')
        #if ( count == 100):
        #    break

#print (listTokens)
print (diclabel)
print ("Max vocab length : " ,maxvocab)


# In[3]:


from sklearn import preprocessing

labelen = preprocessing.LabelEncoder()
le = labelen.fit(listlabels)
translabel = le.transform(listlabels)
print (translabel)


# In[ ]:


tmp_file   = get_tmpfile("test_word2vec.txt")
glove2word2vec("glove.twitter.27B.200d.txt", tmp_file)
word_vectors = KeyedVectors.load_word2vec_format(tmp_file)
embedding_matrix = np.zeros((len(listTokens) , maxvocab, EMBEDDING_DIM))


# In[ ]:


print (embedding_matrix.shape)


# In[ ]:


for i, sample in enumerate(listTokens):
    for j, word in enumerate(sample):
        if word in word_vectors.vocab:
            embedding_vector = word_vectors[word]
            #embedding_matrix[i] = embedding_vector
            embedding_matrix[i, j] = embedding_vector


# In[ ]:


first_embedding =  np.sum(embedding_matrix[0], axis=1)


# In[ ]:


print (embedding_matrix.shape[0])


# In[ ]:


sentence_embedding = np.zeros((len(listTokens) ,EMBEDDING_DIM))


# In[ ]:


for idx in range(embedding_matrix.shape[0]):
    print(idx, embedding_matrix[idx].shape)
    cur = embedding_matrix[idx]
    first_embedding =  (np.sum(cur, axis=0))
    sentence_embedding[idx] = first_embedding/maxvocab
    print (idx, sentence_embedding[idx].shape, sentence_embedding[idx])


# In[ ]:


tokenizer = Tokenizer(num_words=MAX_NB_WORDS)


# In[ ]:


tokenizer.fit_on_texts(listTokens)

