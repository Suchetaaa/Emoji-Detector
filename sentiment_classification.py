
# coding: utf-8

# In[107]:


import nltk
import keras
import numpy as np
import csv
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[108]:


path = r'D:\soc-Emoji Detection\sentiment_data.csv'


# In[109]:


file = open(path)


# In[110]:


data = csv.reader(file)
data = list(data)


# In[111]:


text = []
labels = [] 

for i in range(100000):
    
    text.append(data[i][5].lower())
    labels.append(data[i][0])


# In[137]:


tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)

word_index = tokenizer.word_index


# In[113]:


print('Found %s unique tokens.' % len(word_index))


# In[114]:


max_len = 0;
for i in sequences:
    if len(i)>max_len:
        max_len = len(i)


text_data = pad_sequences(sequences, maxlen=max_len)


# In[115]:


embedding_index = {}
embedding_file = open(r'D:\soc-Emoji Detection\glove.6B.300d.txt' , encoding = 'utf8')
for line in embedding_file:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:] , dtype = 'float32')
    embedding_index[word] = vector
embedding_file.close()


        
    

    
    
    


# In[116]:


embedding_matrix = np.zeros((len(word_index)+1 , 300))

for word,i in word_index.items() :
    
    embedding_vector = embedding_index.get(word)
    
    if embedding_vector is not None:
        
        embedding_matrix[i] = embedding_vector




# In[117]:


from keras.layers import Embedding
from keras.models import Model
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.initializers import glorot_uniform



# In[118]:


embedding_layer = Embedding (len(word_index)+1 , 300 , weights = [embedding_matrix], input_length = max_len , trainable= False)


# In[119]:


from keras.utils import to_categorical


# In[120]:


labels_data = to_categorical(np.asarray(labels))


# In[121]:


indices = np.arange(text_data.shape[0])
np.random.shuffle(indices)
data = text_data[indices]
labels = labels_data[indices]
from keras.layers import Input


# In[122]:


input_sentence = Input(shape = (max_len,) , dtype = 'int32')


# In[123]:



embeddings = embedding_layer(input_sentence)


# In[124]:



X = LSTM(128 , return_sequences = True)(embeddings)


# In[125]:


X = Dropout(0.5)(X)
X = LSTM(128 , return_sequences = False)(X)
X = Dropout(0.5)(X)


# In[126]:


X =Dense(5)(X)


# In[127]:


X = Activation('softmax')(X)


# In[128]:


model = Model(inputs = input_sentence , outputs = X)


# In[129]:


model.summary()


# In[130]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
data.shape
labels.shape


# In[131]:


model.fit(data, labels, epochs = 5, batch_size = 32, shuffle=True)


# In[ ]:




