
# CNN Text Classification using Keras

keras_cnn_model.py is cnn model in keras for text classification.

keras_train.py creates x_train and y_train. It reads from a directory where each file as some sentences and those sentences 
belong to class - filename. The filename as read in order is the class order.

keras_predict.py predicts the class of new sentences. The order of class in the keras_train while training has to match order in this file.

## To call keras_predict.py

python keras_predict.py <MODEL_PATH> <TOKENIZER_PATH> <SENTENCE_IN_DOUBLE_QUOTES>

I have included different versions of cnn model and keras_train file here,

## v0.1

### keras_train_v0.1.py

Includes KFold cross validation code, classification_report and confusion matrix created on the best model from cross validation.

### keras_cnn_model_v0.1.py

Additional code ( commented right now ) for using Adam optimizer and another layer of convolution with attention layer at the top.

## v0.2

### keras_train_v0.2.py 

Includes what revision 0.1 had, additionally it contains code to include pre-trained glove vector using code from glove_vocab.py

### keras_cnn_model_v0.2.py

It includes layer in following sequence

- Embedding Layer ( includes pretrained glove vector if supplied )
- Convolution 1D  kernel = 1 , stride = 1
- MaxPooling 1D   patch = 3 , stride = 1
- Dropout
- Convolution 1D  kernel = 2 , stride = 1
- MaxPooling 1D   patch = 2 , stride = 1
- Dropout
- Dense           256 as output 
- Dropout
- Dense ( final layer ) 6 as output

### In my dataset, I was able to reach max validation accuracy of 0.8888888955116272

