
Text Classification using Keras

keras_cnn_model.py is cnn model in keras , it uses 1 conv layer and 1 max pooling layer. There is only one filter used. Right now keras does
not have support for multiple conv+maxpooling layer working on same input. 

keras_train.py creates x_train and y_train. It reads from a directory where each file as some sentences and those sentences 
belong to class - filename. The filename as read in order is the class order.

keras_predict.py predicts the class of new sentences. The order of class in the keras_train while training has to match order in this file.

To call keras_predict.py

python keras_predict.py <MODEL_PATH> <TOKENIZER_PATH> <SENTENCE_IN_DOUBLE_QUOTES>

