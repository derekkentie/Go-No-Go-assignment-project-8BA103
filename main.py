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
    X = np.array(X)
    y = np.array(y)
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape, '\n')
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
        n_layers = len(self.weights)
        print(n_layers)
        for i in range(n_layers):
            W = self.weights[i]
            b = self.biases[i]
            
            # compute pre-activation
            print(f"layer {i+1}")
            print("input", activation.shape, "weights", np.array(W).shape, "biases", np.array(b).shape, b)
            z = activation @ W + b
            print(np.array(z).shape)

            z_values.append(z)
            print(f"activation layer {i+1}", activations[i][:5])
            print(f"z_value layer {i+1}", z_values[i][:5])
            # output layer --> softmax
            if i == n_layers - 1:
                activation = self.softmax(z)
                print("softmax")
            # hidden layers → chosen activation function
            else:
                activation = self.nonlin_selector(z)
                print(self.activation_function)
            activations.append(activation)
        print(f"activation layer {len(activations)}", activations[-1][:5])
        print(len(z_values), len(activations))
        print(sum(activations[-1]))
        print(activations[-1])
        return activations, z_values

    
    def backpropagation(self, activations, z_values, y, learning_rate=0.01):
        y_onehot = self.one_hot_encoding(y)
        predictions = activations[-1]
        error = y_onehot - predictions
        print(y_onehot)
        print(predictions)
        print(error)
        loss = self.categorical_cross_entropy(predictions, y_onehot)
        print("loss", loss)
        accuracy = np.mean(np.argmax(predictions, axis = 1) == np.argmax(y_onehot, axis = 1))
        print("accuracy:", accuracy)
        for layer in range(len(activations), 0, -1):
            delta = error * self.d_nonlin_selector(activations[layer])
            self.weights[layer] = learning_rate * delta
            self.biases[layer] += learning_rate * delta


    def train_model(self, X_train, y_train, alpha = 0.1, learning_rate = 0.01, epochs = 1000):
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
        pass

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

   
        pass


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
    
    def d_relu(self, z):
        if z > 0:
            return 1
        else:
            return 0

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

model = MultiLayerPerceptron(6, 3)
print(model.architecture.layer_sizes)
activations, z_values = model.forward(X_train)
model.backpropagation(activations, z_values, y_train)