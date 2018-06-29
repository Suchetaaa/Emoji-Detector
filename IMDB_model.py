import numpy as np 
from numpy import genfromtxt
import pandas as pd 
import os 
import nltk 
import codecs
import csv
import string
from nltk.stem.porter import PorterStemmer 
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding


path_data = 'IMDB_reviews.csv'
path_embedding = 'glove.6B.100d.txt'
embedding_dim = 100
max_length = 200
split_percentage = 80 

file = codecs.open(path_data, "r",encoding='utf-8', errors='ignore')
full_data = list(csv.reader(file, delimiter=','))
full_data = np.array(full_data)

#Creating a labels numpy array 
labels = full_data[1:, 3]
labels = (labels == 'pos')
labels = labels.astype(int)

#Separating out the tweets/text 
text_data = full_data[1:, 2]

split = int((split_percentage*full_data.shape[0])/100)

table = str.maketrans('', '', string.punctuation)
porter = PorterStemmer()
lemmatizer = WordNetLemmatizer()

data = []
words = []
for i in range(0, text_data.shape[0]) : 
	words = text_data[i].split()
	stripped_words = [w.translate(table) for w in words]
	lowercase_words = [w.lower() for w in stripped_words]
	stemmed_words = [porter.stem(w) for w in lowercase_words]
	lemmatized_words = [lemmatizer.lemmatize(w) for w in stemmed_words]
	text = " ".join(lemmatized_words)
	data.append(text)

	
data = np.asarray(data)

vocab_length = 20000
col = ['text']
df = pd.DataFrame(data = data, columns = col)
tokenizer = Tokenizer(num_words = vocab_length)
tokenizer.fit_on_texts(df['text'])
num_seq = tokenizer.texts_to_sequences(df['text'])
final_seq = pad_sequences(num_seq, maxlen= max_length)

training_seq = final_seq[:split, :]
test_seq = final_seq[split:, :]

training_labels = labels[:split]
test_labels = labels[split:]

#converting the embedding into a dict 
embedding_dict = dict()
with open(path_embedding, 'r') as f : 
	for line in f : 
		values = line.split()
		embedding_dict[str(values[0])] = np.asarray(values[1:], dtype = 'float32')

embedding_matrix = np.zeros((vocab_length, embedding_dim))
for word, index in tokenizer.word_index.items() : 
	if index > vocab_length - 1 : 
		break 
	embedding_vector = embedding_dict.get(word)
	if embedding_vector is not None : 
		embedding_matrix[index] = embedding_vector

model = Sequential()
model.add(Embedding(vocab_length, embedding_dim, input_length=max_length, weights=[embedding_matrix], trainable=False))
model.add(LSTM(embedding_dim, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(training_seq, np.array(training_labels), validation_split=0.4, epochs=30)
model.summary()
scores = model.evaluate(test_seq, test_labels)

print ('Test loss :', scores[0])
print ('Test accuracy :', scores[1])









	







