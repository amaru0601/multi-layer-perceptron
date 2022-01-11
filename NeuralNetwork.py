import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

LEARNING_RATE = 0.01
EPOCHS = 20
np.random.seed(7) #agregamos una semilla

df = pd.read_csv('data/sign_mnist_train.csv')

imgs =df.iloc[:,1:] #separamos las imagenes de los labels
y = df.iloc[:,0:1] #capturamos los labels
y = y.to_numpy()
unique, counts = np.unique(y, return_counts=True)
#print(len(unique))
#print(np.asarray((unique, counts)).T)

#no cases for 9=J or 25=Z because of gesture motions
y_train = np.zeros((len(y), 26))
for i, y in enumerate(y):
    y_train[i][y] = 1
#print(y.shape)
#print(y_train)
#regularizamos los valores del train set
mean = np.mean(imgs) 
stdev = np.std(imgs)
x_train = (imgs - mean) / stdev
x_train = x_train.to_numpy()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(784, 3)
layer1.forward(x_train)

print(layer1.output[:5])

class Activation_ReLu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

activation1 = Activation_ReLu()
activation1.forward(layer1.output)
print(activation1.output[:5])
