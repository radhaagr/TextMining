#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint


# In[3]:


debug = 1
os.listdir(os.getcwd())

EMBEDDING_DIM=200
VALIDATION_SPLIT = 0.1


# In[4]:


input_file = 'tweet_remove_blank.csv'


# In[5]:


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
        finaltoken = " ".join(finaltoken)

        listTokens.append(finaltoken )
        listlabels.append(label)
        #print( "tweet set : [ " , count ," ] " ,  finaltoken )
        #print('')
        #if ( count == 100):
        #    break

print (diclabel)
print ("Max vocab length : " ,maxvocab)


# In[6]:


from sklearn import preprocessing
labelen = preprocessing.LabelEncoder()
le = labelen.fit(listlabels)
translabel = le.transform(listlabels)
print (translabel)


# In[7]:


tmp_file   = get_tmpfile("test_word2vec.txt")
glove2word2vec("glove.twitter.27B.200d.txt", tmp_file)
word2vec = KeyedVectors.load_word2vec_format(tmp_file)
print('Found %s word vectors of word2vec' % len(word2vec.vocab))


# In[8]:


MAX_NB_WORDS = 200000
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(listTokens)
sequences_1 = tokenizer.texts_to_sequences(listTokens)
word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))
data_1 = pad_sequences(sequences_1, maxlen=maxvocab)


# In[9]:




labels = np.array(translabel)

print('Preparing embedding matrix')
nb_words = min(MAX_NB_WORDS, len(word_index))+1


# In[10]:


embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)

#print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0)


# In[11]:


perm = np.random.permutation(len(data_1))
idx_train = perm[:int(len(data_1)*(1-VALIDATION_SPLIT))]
idx_val = perm[int(len(data_1)*(1-VALIDATION_SPLIT)):]

data_1_train = np.vstack(data_1[idx_train])
labels_train = labels[idx_train]

data_1_val = np.vstack(data_1[idx_val])
labels_val = labels[idx_val]
re_weight = True
weight_val = np.ones(len(labels_val))
if re_weight:
    weight_val *= 0.472001959
    weight_val[labels_val==0] = 1.309028344


# In[63]:



from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

lstm_out = 196
num_dense = np.random.randint(100, 150)
rate_drop_dense = 0.15 + np.random.rand() * 0.25

embedding_layer =  Embedding(nb_words, EMBEDDING_DIM,weights=[embedding_matrix],  input_length=maxvocab)
lstm_layer = LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2)
sequence_1_input = Input(shape=(maxvocab,), dtype='int32')
embedded_sequences_1 = embedding_layer(sequence_1_input)
x1 = lstm_layer(embedded_sequences_1)

merged  = Dropout(rate_drop_dense)(x1)
merged = BatchNormalization()(merged)

merged = Dense(num_dense, activation='relu')(merged)
merged = Dropout(rate_drop_dense)(merged)
merged = BatchNormalization()(merged)

preds = Dense(6, activation='softmax')(merged)

model = Model(inputs=[sequence_1_input],  outputs=preds)

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics =['categorical_accuracy'])
print(model.summary())


# In[64]:


from sklearn.model_selection import train_test_split
from keras.utils import np_utils

#labels  = np_utils.to_categorical(labels, 6)
print (translabel.shape)
getar = np.array(translabel)
print (getar.shape)
newl = np_utils.to_categorical(getar, 6)
print (newl.shape)


# In[65]:


X_train, X_test, Y_train, Y_test = train_test_split( data_1 ,newl, test_size = 0.33, random_state = 42)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


# In[17]:


#batch_size = 32

#model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, verbose = 2)


# In[66]:


hist = model.fit(X_train, Y_train,         validation_data=(X_test, Y_test ),         epochs=5, batch_size=50, shuffle=True )


bst_val_score = min(hist.history['val_loss'])


# In[67]:


test_labels = []
label_file = 'test.labels'
with open(label_file,  encoding='utf-8') as f:
    for i,line in enumerate(f):
        test_labels.append(line.strip())
          
print ("labels" ,len(test_labels))


# In[68]:


testsequence = []
test_file = "test.csv"
count = 0
with open(test_file,  encoding='utf-8') as f:
    for i,line in enumerate (f):
        text = line.strip()
        htmltext = removehtmlentities(text)
        urltext  = removeurl(htmltext)
        attext = removeat(urltext)
        triggertext =  removetriggerword(attext)
        expandabbrevation(triggertext)
        finaltoken = removestopword(tokenizeString(triggertext))
        curtotal = len(finaltoken)
        testsequence.append(finaltoken )
        #if( count == 100 ):
        #    break
        #count = count + 1
#print ("Testdata" , testsequence)


# In[69]:


tokenizer.fit_on_texts(testsequence)
test_sequences_1 = tokenizer.texts_to_sequences(testsequence)
test_data_1 = pad_sequences(test_sequences_1, maxlen=maxvocab)


# In[70]:


preds = model.predict(test_data_1, batch_size=8192, verbose=1)
print ( preds.argmax(axis=1))
result = preds.argmax(axis=1)


# In[71]:


from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
labelen = preprocessing.LabelEncoder()
le = labelen.fit(test_labels)
#print (list(le.classes_))
trans_testlabel = le.transform(test_labels)

x = confusion_matrix(trans_testlabel, result)
print (x) 


# In[72]:


from sklearn.metrics import classification_report

print(classification_report(trans_testlabel, result, target_names=list(le.classes_)))


# In[59]:


from sklearn import preprocessing
labelen = preprocessing.LabelEncoder()
le = labelen.fit(test_labels)
#print (list(le.classes_))
trans_testlabel = le.transform(test_labels)
#print (trans_testlabel)
#print (trans_testlabel.shape)
category_test_label = np_utils.to_categorical(trans_testlabel, 6)

#print (category_test_label.shape)
#print (list(le.inverse_transform([1,0, 2, 3,4,5])))


# In[45]:


#print  (test_labels[0:10])
#print (trans_testlabel[0:10] )
#print ( category_test_label[0:10])


# In[ ]:


from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred, target_names=target_names))


# In[61]:


scores = model.evaluate(test_data_1, category_test_label, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))


# In[ ]:


from sklearn.metrics import confusion_matrix

x = confusion_matrix(y_true, y_pred)

