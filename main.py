import numpy as np
import pandas as pd

# The purpose of the model is to accurately categorise the sleep quality of each sample in the categories: bad, medium and good,
# based on the variables: Age, sleep duration, study hours, screen time, caffeine intake and physical activity.
# This means that we 6 variables for the input layer (+1 bias node), and 3 categories for the output layer.

def data_extraction_csv(csv_file):
    """ 
    This function extracts data from a csv file and splits the inputs and outputs in different arrays called X and y,
    assuming that the csv file has a header in the first row and the output variables stored in the last column 
    """
    data = np.array(pd.read_csv(csv_file, sep=','))
    # creating the lists from the input variables and labels
    X = []
    y = []

    for line in data:
        X.append(line[:-1])
        y.append(line[-1])

    # converting the lists to numpy arrays
    X = np.array(X, dtype=float)
    y = np.array(y)
    return X, y

class Architecture:
    """ 
    Architecture is used to save every connection between layers in the multilayer perceptron. 
    It's advantage is that the editing of the model's architecture has now become extremely versatile, 
    and eliminates the need of hard coding every forward and backward propagation between layers. 
    """
    
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input = input_size
        self.hidden = list(hidden_sizes)
        self.output = output_size

    @property #makes the layer_sizes function a property of the class Architecture, which eliminates the need of () after every call to this function
    def layer_sizes(self):
        return [self.input] + self.hidden + [self.output]


class MultiLayerPerceptron:
    def __init__(self, n_features, n_classes, hidden_size = (4,4), activation_function = "ReLU"):

        #assigning all the values 
        self.input_size = n_features
        self.hidden_size = hidden_size
        self.output_size = n_classes
        self.activation_function = activation_function
        
        #connecting the Architecture class as an object, this will later be used to assign a parameter to every connection between the nodes
        self.architecture = Architecture(
            input_size=self.input_size,
            hidden_sizes=self.hidden_size,
            output_size=self.output_size
        )
        #initializing the parameters
        self.initialise_parameters()
        self.n_layers = len(self.weights)

    def initialise_parameters(self):
        """ 
        Here we use the output of the Architecture class to assign 
        a weight/parameter to every connection between the nodes.
        """
        layer_sizes = self.architecture.layer_sizes
        self.weights = []
        self.biases = []

        #creating the weights and biases
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]

            #Chosing between initialization methods based on hidden layer activation function
            if self.activation_function == "ReLU":
                initialization_method = self.he_init(in_dim)
            elif self.activation_function in ["sigmoid", "tanh"]:
                initialization_method = self.xavier_init(in_dim)
            else:
                raise ValueError("Activation function must be 'ReLU' (standard), 'sigmoid' or 'tanh'.")

            #apply the initialization method to avoid vanishing or exploding weights
            W = np.random.randn(in_dim, out_dim) * initialization_method 
            b = np.zeros((1, out_dim))

            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        """
        Perform a forward pass through the network.
        Returns:
            activations: list of activations per layer (including input layer)
            zs: list of pre-activation z-values per layer
        """
        activations = [X] # a0 = input layer
        z_values = [] # to store z-values
        activation = X

        for i in range(self.n_layers):
            W = self.weights[i]
            b = self.biases[i]
            
            # compute pre-activation
            z = activation @ W + b
            z_values.append(z)

            # output layer --> softmax
            if i == self.n_layers - 1:
                activation = self.softmax(z)
            # hidden layers → chosen activation function
            else:
                activation = self.nonlin_selector(z)
            activations.append(activation)
        return activations, z_values

    
    def backpropagation(self, activations, z_values, y, learning_rate=0.01):
        y_onehot = self.one_hot_encoding(y)
        n_samples = len(y)
        predictions = activations[-1]
        delta = predictions - y_onehot

        weight_gradients = [np.zeros(weight.shape) for weight in self.weights]
        bias_gradients = [np.zeros(bias.shape) for bias in self.biases]

        bias_gradients[-1] = np.sum(delta, axis=0, keepdims=True) / n_samples
        weight_gradients[-1] = activations[-2].T @ delta / n_samples

        #calculating the gradients for the weights and biases
        for layer in range(self.n_layers-2, -1, -1):
            delta = (delta @ self.weights[layer+1].T) * self.d_nonlin_selector(z_values[layer])
            weight_gradients[layer] = activations[layer].T @ delta / n_samples
            bias_gradients[layer] = np.sum(delta, axis=0, keepdims=True) / n_samples

        #updating the weights and biases
        for layer in range(self.n_layers):
            self.weights[layer] -= learning_rate * weight_gradients[layer]
            self.biases[layer] -= learning_rate * bias_gradients[layer]


    def train_model(self, X_train, y_train, learning_rate = 0.001, epochs = 1000):
        history = {
            "loss": [],
            "accuracy": []
        }
        count = 0
        y_onehot = self.one_hot_encoding(y_train)
        for epoch in range(epochs):
            activations, z_values = self.forward(X_train)
            self.backpropagation( activations, z_values, y_train, learning_rate)

            #keeping track of loss and accuracy
            predictions = activations[-1]
            loss = self.categorical_cross_entropy(predictions, y_onehot)
            accuracy = np.mean(np.argmax(predictions, axis = 1) == np.argmax(y_onehot, axis = 1))
            history["loss"].append(loss)
            history["accuracy"].append(accuracy)

            if epoch == count+25:
                print(f"epoch: {epoch}/{epochs}")
                print("loss", loss)
                print("accuracy:", accuracy)
                count = epoch

        model_params = {
            "weights": self.weights,
            "biases": self.biases,
            "train history":  history
        }

        return model_params

    def predict(self, X_test, model_params):
        self.weights = model_params["weights"]
        self.biases = model_params["biases"]

        activations, _ = self.forward(X_test)
        return np.argmax(activations[-1], axis=1)


    #activation functions
    def nonlin_selector(self, z):
        if self.activation_function.lower() == "relu":
            return self.relu(z)
        elif self.activation_function.lower() == "sigmoid":
            return self.sigmoid(z)
        elif self.activation_function.lower() == "tanh":
            return self.tanh(z)
        else:
            ValueError(
                f"{self.activation_function} is not valid activation function for this model, choose between ReLU, sigmoid or tanh."
            )

    #optional activation function in hidden layer
    def sigmoid(self, z): 
        z = np.array(z, dtype=float) 
        return 1 / (1 + np.exp(-z))
    
    def tanh(self, z):
        return np.tanh(z)
    
    def relu(self, z): 
        return np.maximum(0, z)
    
    #softmax is used in the output layer to enable the model to classify with multiple classes
    def softmax(self, z): 
        z_softmax = []
        for observation in z: #performing softmax per observation
            observation = np.array(observation, dtype=float) #making sure that the observations are vectors
            e_observation = np.exp(observation - np.max(observation)) #exponentiation
            norm_e_observation = (e_observation / np.sum(e_observation)) #normalization to get softmax
            z_softmax.append(norm_e_observation)
        return np.array(z_softmax)

    #derivative activation functions
    def d_nonlin_selector(self, z):
        if self.activation_function == "ReLU":
            return self.d_relu(z)
        elif self.activation_function == "sigmoid":
            return self.d_sigmoid(z)
        elif self.activation_function == "tanh":
            return self.d_tanh(z)

    def d_sigmoid(self, z):
        f = self.sigmoid
        return f(z)*(1-f(z))
    
    def d_tanh(self,z):
        f = self.tanh
        return 1- f(z)*f(z)
    """ 
    def d_relu(self, z):
        return (z > 0).astype(float)
    """


    def d_relu(self, z):
        d_relu = []
        for input in z:
            d_relu_part = []
            for value in input:
                if value > 0:
                    d_relu_part.append(1)
                else:
                    d_relu_part.append(0)
            d_relu.append(d_relu_part)
        return np.array(d_relu)

    def d_softmax(self, z):
        f = self.softmax
        return f(z)*(1-f(z))


    #Initialization methods to prevent vanishing or exploding weights
    def xavier_init(self, in_dim):
        return np.sqrt(1 / in_dim) #used for sigmoid or tanh networks
    
    def he_init(self, in_dim):
        return np.sqrt(2 / in_dim) #used for ReLU networks
        
    def one_hot_encoding(self, y):
        y_factorize, labels = pd.factorize(y)
        base_arr = np.zeros((np.array(y).shape[0], len(labels)))
        for observation in range(len(y_factorize)):
            base_arr[observation][y_factorize[observation]] = 1
        y_onehot = base_arr
        return y_onehot
        
    def categorical_cross_entropy(self, y_predict, y):
        y_predict_clipped = np.clip(y_predict, 1e-15, 1 - 1e-15) #clipping the predictions to prevent log(0)
        return -np.sum(y * np.log(y_predict_clipped)) / y.shape[0] #returning mean of sample losses

X_train, y_train = data_extraction_csv("data/train.csv")
X_test, y_test = data_extraction_csv("data/test.csv")
model = MultiLayerPerceptron(6, 3)
print(model.architecture.layer_sizes)
params = model.train_model(X_train, y_train)
model.predict(X_test, params)