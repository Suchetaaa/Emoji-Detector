import numpy as np 
import csv
import pandas as pd
import matplotlib.pyplot as plt 
import math





def converting_into_one_label(dataset):

	
	#extracting column 
	Valence1 = dataset['Valence1']
	Valence2 = dataset['Valence2']
	Arousal1 = dataset['Arousal1']
	Arousal2 = dataset['Arousal2']
	



	#defining region to classify into 4 categories
	label1 = []
	for i in range(len(Valence2)):

		if Valence1[i] <= 5:
			if Arousal1[i] <= 5:
			
				label1.append(0) # here zero resemble SAD
		
		if Valence1[i] > 5:
			if Arousal1[i] <= 5:

				label1.append(1) #here One resemble PEACEFUL

		if  Valence1[i] <= 5 :
			if Arousal1[i] > 5:

				label1.append(2) #here Two resemble Angry

		if Valence1[i] > 5:
			if Arousal1[i] > 5:

				label1.append(3) #here Three resemble Happy

		i = i+1
	#end

	#stackinng and conerting into column
	emotions1 = np.asarray(label1)
	emoji1 = np.vstack(emotions1) 

	
	

	#same processor defining and converting into column label2
	label2 = []
	for i in range(len(Valence2)):

		if Valence2[i] <= 5:
			if Arousal2[i] <= 5:
			
				label2.append(0) # here zero resemble SAD
		
		if Valence2[i] > 5:
			if Arousal2[i] <= 5:

				label2.append(1) #here One resemble PEACEFUL

		if  Valence2[i] <= 5 :
			if Arousal2[i] > 5:

				label2.append(2) #here Two resemble Angry

		if Valence2[i] > 5:
			if Arousal2[i] > 5:

				label2.append(3) #here Three resemble Happy

		i = i+1
	#end


	emotions2 = np.asarray(label2)
	emoji2 = np.vstack(emotions2) 




	#Taking average and greatest integer  of label1 and label2 and  storing in y_data
	emoji = []
	for i in range(len(emoji1)):
		avg = np.average((emoji1[i], emoji2[i]))
		
		emoji.append(avg)
		i = i+1
	#end

	y_data  = np.asarray(emoji)
	y_data = np.vstack(y_data)

	for i  in range(len(y_data)):
		
		y_data[i]  = math.ceil(y_data[i])
		i = i+1
	#end
		
	#adding new column to the existing csv file as Lables which store y_data
	df = pd.read_csv("dataset-fb-valence-arousal-anon.csv")
	df['labels'] = y_data
	#print(df)
	df.to_csv('dataset-fb-valence-arousal-anon.csv')


	#print df[5]

	#print df

	keep_df = ["Anonymized Message" , "labels"]

	new_dataset = df[keep_df]

	new_dataset.to_csv("new_dataset.csv", index = False)

	print new_dataset
	return new_dataset

data = pd.read_csv("dataset-fb-valence-arousal-anon.csv")
x = converting_into_one_label(data)



