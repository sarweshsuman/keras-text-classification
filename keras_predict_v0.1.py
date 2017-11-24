from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import sys

"""
sys.argv[1] = saved model full path
sys.argv[2] = pickled tokenizer full path
sys.argv[3] = text to classifiy
"""

class_category = ['UX ANALYST','SYS ANALYST','CONFIGURATION','BIZ ANALYST','DATA ANALYST','SECURITY'];

if len(sys.argv) != 4:
	print("Parameters missing")
	sys.exit(1)
model = load_model(sys.argv[1])
tokenizer = pickle.load(open(sys.argv[2],'rb'))

x_pred = tokenizer.texts_to_sequences([sys.argv[3]])
x_pred = pad_sequences(x_pred,maxlen=50) # Max len hardcoded here, has to be parameterized in case of production version
result = model.predict(x_pred)
print(class_category[result.argmax()])
