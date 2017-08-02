from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
import sys
import pickle
import numpy as np
from keras_cnn_model import create_model
from keras.utils import to_categorical

texts=[]
labels=[]

def read_inputs(folder_name):
	global texts
	global labels
	dirs=os.listdir(folder_name)
	class_id=0
	for fn in dirs:
		print("Processing {}".format(fn))
		full_path = os.path.join(folder_name,fn)
		fh=open(full_path)
		lines=fh.readlines()
		fh.close()
		texts = texts+lines
		[labels.append(class_id) for x in lines]
		class_id += 1

if __name__ == '__main__':
	read_inputs('./data')
	tokenizer = Tokenizer(num_words=500)
	tokenizer.fit_on_texts(texts)
	sequences = tokenizer.texts_to_sequences(texts)
	word_index = tokenizer.word_index
	vocab_size = len(word_index)
	data=pad_sequences(sequences,maxlen=50)
	print("Length of training data {}".format(len(data)))
	print("Shape of data {}".format(data.shape))
	indices = np.arange(data.shape[0])
	np.random.shuffle(indices)
	print("Indices {}".format(indices))
	data = data[indices]
	labels = to_categorical(np.asarray(labels))  # this converts [0,0,1,1] to [[1..],[1...],[0 1 0...]..]
	print(labels)
	labels = labels[indices]
	model = create_model(vocab_size,100,50,(3,),256,0.5) # As keras does not have support for multi filters in cnn on same output from embedding layer hence proceeding with one layer of cnn with one filter
	""" Ready to train """
	model.fit(data,labels,epochs=500,batch_size=16)
	model.save('./keras_saved_model/intent_model.h5')
	pickle.dump(tokenizer,open('./keras_saved_model/tokenizer.p','wb'))
