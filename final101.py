import numpy as np
np.random.seed(0)
import csv
import nltk
import pandas as pd 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer 

#print (" you bro")

data = pd.read_csv("dataset-fb-valence-arousal-anon.csv")
#print len(data)

text = data['Anonymized Message']
text = list(text)
print text