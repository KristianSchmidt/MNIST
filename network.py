from keras.models import Sequential
from keras.layers import Dense, Activation
import keras.utils
import pandas as pd

train_file = '.\\Data\\train.csv'
traindf = pd.read_csv(train_file)
x_train = traindf.as_matrix(columns=[ 'pixel' + str(x) for x in range(784)]).astype('float32')

test_file = '.\\Data\\test.csv'
testdf = pd.read_csv(test_file)
x_test = testdf.as_matrix(columns=None).astype('float32')

# Normalization of the input data. The values go from 0 to 255, so this transformation will keep them
# between 0.0 and 1.0
x_train /= 255.0
x_test /= 255.0

y_train = traindf.as_matrix(columns=['label'])
y_train = keras.utils.np_utils.to_categorical(y_train, 10)

model = Sequential(
    [
     Dense(32, input_dim=784, activation='relu'),
     Dense(15, activation='relu'),
     Dense(10, activation='softmax')
    ]
)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, nb_epoch=15, batch_size=32)

# The model has been trained. Now we can generate our guesses for the test data
predictions = model.predict_classes(x_test)

submission = pd.DataFrame(data={'ImageId': [x for x in range(1,28001)], 'Label': predictions})
submission.to_csv('submission.csv', index=False)