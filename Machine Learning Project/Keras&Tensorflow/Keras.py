import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.regularizers import l1, l2

# Import data
dataframe = pd.read_csv('CE 475 Fall 2018 Project Data - data.csv')
dataset = dataframe.values

dataframetest = pd.read_csv('CE 475 Fall 2018 Test-data.csv')
datasettest = dataframetest.values


# Random shuffle the dataset
max_num_X = np.amax(dataset[:, 0:-1])
max_num_Y = np.amax(dataset[:, -1])

max_num_X_test = np.amax(20)
max_num_Y_test = np.amax(datasettest[:, -1])

# Split into train and test sets
train_size = 100
test_size = 20
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]


# split labels and datas
def create_dataset(dataset):
    dataX, dataY = [], []
    for i in range(len(dataset)):
        rowX = dataset[i, 0:-1].astype('int')
        dataX.append(rowX)
        dataY.append(dataset[i, -1])
    return dataX, dataY


trainX, trainY = create_dataset(train)
testX, testY = create_dataset(test)


# reshape trainX and testX to be [samples, time steps, features]
samples_train = len(trainX)
samples_test = len(testX)
time_steps = len(trainX[1])
trainX_input = np.reshape(trainX, (samples_train, time_steps, 1))
testX_input = np.reshape(testX, (samples_test, time_steps, 1))

# one hot encode the output variable
output_size =  max_num_Y + 1
trainY_cat = np_utils.to_categorical(trainY, output_size)
testY_cat = np_utils.to_categorical(testY,  output_size)

# create the model
input_dimension = 1
model = Sequential()
model.add(LSTM(16, input_shape=(time_steps, input_dimension), dropout_W=0.2, dropout_U=0.2))
model.add(Dense(output_size, activation='softmax', W_regularizer=l2(0.01)))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
print ('Toy LSTM model created...')

# training
print ('Training...')
early_stopping = EarlyStopping(monitor='val_loss', patience=5000)
model.fit(trainX_input, trainY_cat, nb_epoch=5000, batch_size = 10, verbose=2, validation_split=0.2, callbacks=[early_stopping])

# Save model and weight
model_json = model.to_json()
with open("Toy_LSTM_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("Toy_LSTM_model_weights.h5", overwrite= True)
print ("Toy LSTM model and weights saved..")

# testing
print ('Testing on test data...')
scores = model.evaluate(testX_input, testY_cat, verbose=2)
print ('Test accuracy on test data: {}'.format(scores[1]))

print ('Testing on training data...')
scores = model.evaluate(trainX_input, trainY_cat, verbose=2)
print ('Test accuracy on training data: {}'.format(scores[1]))

# Prediction
print ('Predicting on testing data...')
for row in testX:
    x = np.reshape(row, (1, len(row), 1))
    prediction = model.predict(x, verbose=0)
    y_hat = np.argmax(prediction)
    print ((row*max_num_X).astype("int")/max_num_X, "-->" ,y_hat)

print ('Predicting on training data...')
for row in trainX:
    x = np.reshape(row, (1, len(row), 1))
    prediction = model.predict(x, verbose=0)
    y_hat = np.argmax(prediction)
    print ((row*max_num_X).astype("int")/max_num_X, "-->" ,y_hat)