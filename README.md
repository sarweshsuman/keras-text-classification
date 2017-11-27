
# CNN Text Classification using Keras

keras_cnn_model.py is cnn model in keras for text classification.

keras_train.py creates x_train and y_train. It reads from a directory where each file as some sentences and those sentences 
belong to class - filename. The filename as read in order is the class order.

keras_predict.py predicts the class of new sentences. The order of class in the keras_train while training has to match order in this file.

## To call keras_predict.py

python keras_predict.py <MODEL_PATH> <TOKENIZER_PATH> <SENTENCE_IN_DOUBLE_QUOTES>

I have included different versions of cnn model and keras_train file here,

## v0.1

keras_train_v0.1.py --> includes KFold cross validation code, classification_report and confusion matrix created on the best model from cross validation.
keras_cnn_model_v0.1.py  --> additional code ( commented right now ) for using Adam optimizer and another layer of convolution with attention layer at the top.



