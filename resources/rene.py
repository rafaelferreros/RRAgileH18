# Importing Keras Sequential Model
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy

# Initializing the seed value to a integer.
seed = 7

numpy.random.seed(seed)

# Loading the data set (PIMA Diabetes Dataset)
#dataset = numpy.loadtxt('datasets/pima-indians-diabetes.csv', delimiter=",")
data_file =  '../dataset/hackaton_training_v1.csv'
dataset = pd.read_csv(data_file, usecols=['v_0', 'v_1', 'v_2', 'v_3', 'v_4',
                                           'v_5', 'v_6', 'v_7', 'v_8', 'v_9',
                                           'v_10', 'v_11', 'v_12'])

#print(dataset.head())

# Loading the input values to X and Label values Y using slicing.

X = dataset.loc[:, ['v_2','v_4','v_6','v_9']]
print(X.head())
Y = dataset.loc[:, 'v_10']
print(Y.head())

# Initializing the Sequential model from KERAS.
model = Sequential()

# Creating a 16 neuron hidden layer with Linear Rectified activation function.
model.add(Dense(16, input_dim=4, init='uniform', activation='relu'))

# Creating a 8 neuron hidden layer.
model.add(Dense(8, init='uniform', activation='relu'))

# Adding a output layer.
model.add(Dense(1, init='uniform', activation='sigmoid'))



# Compiling the model
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
# Fitting the model
model.fit(X, Y, epochs=100, batch_size=30)

scores = model.evaluate(X, Y)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))