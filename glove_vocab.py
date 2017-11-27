import os
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

class Glove:
	def __init__(self,glove_path,embedding_size,load_from_exiting_path=None):
		if load_from_exiting_path != None:
			if os.path.isdir(load_from_exiting_path) == False:
				raise Exception("path {} is not found".format(load_from_exiting_path))
			embeddings_index_path = os.path.join(load_from_exiting_path,'embedding_index.bin')
			embedding_matrix_path = os.path.join(load_from_exiting_path,'embedding_matrix.bin')
			if os.path.isfile(embeddings_index_path) == False or os.path.isfile(embedding_matrix_path) == False:
				raise Exception("file {} and {} not found".format(embeddings_index_path,embedding_matrix_path))
			self.embeddings_index=pickle.load(embeddings_index_path)
			self.embedding_matrix=pickle.load(embedding_matrix_path)
			return
		if os.path.isfile(glove_path) == False:
			raise Exception("glove file {} file not found".format(glove_path))
		self.glove_path=glove_path
		self.embedding_size=embedding_size
		self.embeddings_index={}
		f = open(os.path.join(self.glove_path))
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			self.embeddings_index[word] = coefs
		f.close()
	def get_embedding_index(self):
		return self.embeddings_index
	def create_embedding_matrix(self,word_index):
		""" word_index is tokenizer.word_index and tokenizer is from keras.preprocessing.text import Tokenizer """
		self.embedding_matrix = np.zeros((len(word_index)+1, self.embedding_size))
		for word, i in word_index.items():
			embedding_vector = self.embeddings_index.get(word)
			if embedding_vector is not None:
				# words not found in embedding index will be all-zeros.
				self.embedding_matrix[i] = embedding_vector
	def get_vector(self,word_id):
		""" word_id is number assigned to the word when create_embedding_matrix is called """
		return self.embedding_matrix.get(word_id)
	def store(self,store_path):
		if os.path.isdir(store_path) == False:
			raise Exception("unable to save to {} , dir does not exists".format(store_path))
		pickle.dump(self.embeddings_index,open(os.path.join(store_path,'embedding_index.bin'),'wb'))
		pickle.dump(self.embedding_matrix,open(os.path.join(store_path,'embedding_matrix.bin'),'wb'))

class Vocabulary:
	def __init__(self,max_num_words,max_sequence_length,load_from_exiting_path=None):
		if load_from_exiting_path != None:
			if os.path.isdir(load_from_exiting_path) == False:
				raise Exception("path {} is not found".format(load_from_exiting_path))
			vocabulary_path = os.path.join(load_from_exiting_path,'vocabulary.bin')
			if os.path.isfile(vocabulary_path) == False:
				raise Exception("file {} and {} not found".format(embeddings_index_path,embedding_matrix_path))
			vocab = pickle.load(open(vocabulary_path,'rb'))
			self.max_num_words=vocab.max_num_words
			self.max_sequence_length=vocab.max_sequence_length
			self.tokenizer=vocab.tokenizer
			self.sequences=vocab.sequences
			self.data=vocab.data
			return
		self.max_num_words=max_num_words
		self.max_sequence_length=max_sequence_length
		self.tokenizer = Tokenizer(num_words=max_num_words)
	def get_word_index(self):
		return self.tokenizer.word_index
	def fit_and_pad(self,texts):
		""" called while training """
		self.tokenizer.fit_on_texts(texts)
		self.sequences = self.tokenizer.texts_to_sequences(texts)
		self.data=pad_sequences(self.sequences,maxlen=self.max_sequence_length)
		return self.data
	def get_padded_sequences(self,texts):
		""" called while inference """
		self.sequences = self.tokenizer.texts_to_sequences(texts)
		self.data=pad_sequences(self.sequences,maxlen=self.max_sequence_length)
		return self.data
	def store(self,store_path):
		if os.path.isdir(store_path) == False:
			raise Exception("unable to save to {} , dir does not exists".format(store_path))
		pickle.dump(self,open(os.path.join(store_path,'vocabulary.bin'),'wb'))

def read_class_files_for_training(class_info_file,class_data_directory):
	texts = []  # list of text samples
	labels_index = {}  # dictionary mapping label id to name
	labels = []  # list of label ids
	if os.path.isfile(class_info_file) == False:
		raise Exception("file {} not found".format(class_info_file))
	fh=open(class_info_file)
	classes = fh.readlines()
	fh.close()
	
	idx=0
	for cls in classes:
		print(idx)
		cls = cls.replace('\n','')
		full_path = os.path.join(class_data_directory,cls)
		if os.path.isfile(full_path) == False:
			raise Exception("file {} not found".format(full_path))
		fh=open(full_path)
		lines=fh.readlines()
		fh.close()
		for ln in lines:
			ln = ln.replace('\n','')
			texts.append(ln)
			labels.append(idx)
		labels_index[idx]=cls
		idx += 1
	labels = to_categorical(np.asarray(labels))
	return (texts,labels,labels_index)
def read_class_file_for_prediction(class_info_file):
	labels_index = {}  # dictionary mapping label id to name
	if os.path.isfile(class_info_file) == False:
		raise Exception("file {} not found".format(class_info_file))
	fh=open(class_info_file)
	classes = fh.readlines()
	fh.close()
	idx=0
	for cls in classes:
		labels_index[idx]=cls
		idx += 1
	return labels_index
def read_training_file_for_retrieval_model(file_name):
	""" 
		Training file is supposed to be of following format 
		line[1]: question | answer | 1   --> for correct answer against the question
		line[2]: question | answer | 0   --> for incorrect answer agains the question
	"""
	fh = open(file_name,'r')
	lines = fh.readlines()
	fh.close()

	questions=[]
	answers=[]
	target=[]
	for ln in lines:
		ln = ln.strip()
		arr=ln.split('|')
		questions.append(arr[0].strip())
		answers.append(arr[1].strip())
		target.append(arr[2].strip())
	return (questions,answers,target)

