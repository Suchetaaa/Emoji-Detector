import numpy as np 
from divide_dataset import converting_into_one_label
np.random.seed(0)
from keras.models import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding 
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
import nltk
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer 
np.random.seed(1)





def word_to_index(word):
	tokenize = Tokenizer()
	tokenize.fit_on_texts(word)
	return tokenizer.word_index 	

def word_to_vec_map(ves):




def sentences_to_indices(X, word_to_index, max_len):
	m = X.shape[0]
	X_indices = np.zeros((m, max_len))

	for i in range(m):
		sentence_words = [w.lower() for w in X[i].spilt()]
		j = 0
		for w in sentence_words:
			X_indices[i,j] = word_to_index[w]
			j += 1

	return X_indices


def  pretrained_embedding_layer(word_to_vec_map, word_to_index):
	vocab_len = len(word_to_index) + 1
	emb_dim = word_to_vec_map["cucumber"].shape[0]
	emb_matrix = np.zeros((vocab_len, emb_dim))

	for word, index in word_to_index.items():
		emb_matrix[index,:] = word_to_vec_map[word]

	embedding_layer = Embedding(vocab_len, emb_dim, trainable = False)

	embedding_layer.build((None,))
	embedding_layer.set_weights([emb_matrix])


	return embedding_layer










def Emojify_V2(input_sape, word_to_vec_map, word_to_index):
	sentence_indices = Input(input_sape, dtype = 'int32')
	embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
	embeddings = embedding_layer(sentence_indices)
	X = LSTM(128, return_sequences = True)(embeddings)
	X = Dropout(0.5)(X)
	X = LSTM(128, return_sequences = False)(X)
	X = Dropout(0.5)(X)
	X = Dense(5)(X)
	X = Activation('softmax')(X)
	model = Model(inputs = sentence_indices, outputs = X)

	

	model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

	return model



X_train = new_dataset["Anonymized Message"]
X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(Y_train_oh, C = 5)

model.fit(X_train_indices, Y_train_oh, epochs = 50, batch_size = 32, shuffle = True)

X_test_indices = sentence_indices(X_test, word_to_index,max_len = maxLen)
Y_test_oh = convert_to_one_hot(Y_test, C = 5)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print()
print("Test accuracy = ", acc)


C = 5
y_test_oh = np.eye(C)[Y_test.reshape(-1)]
X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
pred = model.predict(X_test_indices)
for i in range(len(X_test)):
	x = X_test_indices
	num = np.argmax([pred[i]])
	if(num != Y_test[i]):
		print('Expected emoji' + label_to_emoji(Y_test[i]) + 'prediction:' + X_test[i] + label_to_emoji(num).strip())




x_test = np.array(['not feeling happy'])
X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
print(x_test[0] + ' ' + label_to_emoji(np.argmax(model.predict(X_test_indices))))























