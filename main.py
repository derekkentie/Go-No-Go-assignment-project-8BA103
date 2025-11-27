import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# The purpose of the model is to accurately categorise the sleep quality of each sample in the categories: bad, medium and good,
# based on the variables: Age, sleep duration, study hours, screen time, caffeine intake and physical activity.
# This means that we 6 variables for the input layer (+1 bias node), and 3 categories for the output layer.

def sigmoid(z): #sigmoid is used for every node in the hidden layer(s)
     return 1 / (1 + np.exp(-z))

def softmax(z): #softmax is used in the output layer to make a enable the model to classify with multiple classes
    return np.exp(-z) / (1 + np.exp(-z))

def augment_matrix(X):
    return np.c_[np.ones((X.shape[0], 1)), X] 

def data_extraction_csv(csv_file):
        """ This function extracts and returns the X_train and y_train for the train_model function"""
        data = [line.strip().split(',') for line in open(csv_file, 'r')] # extracting all the data from the raw csv file and placing it in a list of lists
        data.pop(0) #removing the header
        # creating the lists from the input variables and labels
        X = []
        y = []
        for line in data:
            y.append(line.pop(-1))
            X.append(line)
        # converting the lists to numpy arrays for easier use further on
        X = np.array(X)
        y = np.array(y)
        return X, y

class MultiLayerPerceptron:
    def _init_(self, n_hidden_layers = 6, n_nodes_per_layer = 1, alpha = 0.1, learning_rate = 0.01, epochs = 1000):
        self.theta = None
        self.bias = None
        self.n_hidden_layers = n_hidden_layers
        self.n_nodes_per_layer = n_nodes_per_layer
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
         pass

    def train_model(self, X, y, n_hidden_layers = 6, n_nodes_per_layer = 1, alpha = 0.1, learning_rate = 0.01, epochs = 1000):
        n_samples, n_features = X.shape
        self.theta = np.random.rand(n_features)
        self.bias = np.random.rand()
        print(X)
        print(self.theta, self.bias)
        for epoch in range(epochs):
            prediction = np.dot(X,self.theta)
            error = prediction - y
            gradient_direction = (2/n_samples)*(np.dot(np.transpose(X),error)) # Devided by n_samples to apply MSE instead of RSS
            shrinkage = learning_rate*self.theta*alpha/n_samples
            self.theta = self.theta - learning_rate*gradient_direction - shrinkage
        """
        Trains your ML algorithm on the provided training data.

        Parameters:
            X_train (numpy.ndarray): Training features, shape (n_samples, n_features)
            y_train (numpy.ndarray): Training labels, shape (n_samples,)
            **hyperparams: Algorithm-specific hyperparameters
                        (e.g., learning_rate=0.01, max_iter=1000, k=5, etc.)

        Returns:
            model_params (dict): A dictionary containing all information needed
                                to make predictions later.
                                For example, it might include learned weights,
                                biases, thresholds, or training statistics.
        """

    def predict(self, X_test, model_params):
        """
        Uses the trained parameters to make predictions on new (test) data.

        Parameters:
            X_test (numpy.ndarray): Test features, shape (n_samples, n_features)
            model_params (dict): Dictionary of parameters returned by train_model()

        Returns:
            y_pred (numpy.ndarray): Predicted labels, shape (n_samples,)
        """
        pass
X, y = data_extraction_csv("data\\train.csv")
model = MultiLayerPerceptron()
model.train_model(X, y)
print(augment_matrix(X), X.shape, augment_matrix(X).shape)