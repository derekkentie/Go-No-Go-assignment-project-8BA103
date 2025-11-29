import numpy as np

# The purpose of the model is to accurately categorise the sleep quality of each sample in the categories: bad, medium and good,
# based on the variables: Age, sleep duration, study hours, screen time, caffeine intake and physical activity.
# This means that we 6 variables for the input layer (+1 bias node), and 3 categories for the output layer.

def data_extraction_csv(csv_file):
    """ 
    This function extracts data from a csv file and splits the inputs and outputs in different arrays called X and y,
    assuming that the csv file has a header in the first row and the output variables stored in the last column. 
    """
    data = [line.strip().split(',') for line in open(csv_file, 'r')] # extracting all the data from the raw csv file and placing it in a list of lists
    data.pop(0) #removing the header
    # creating the lists from the input variables and labels
    X = []
    y = []
    for line in data:
        y.append(line.pop(-1))
        X.append(line)
    # converting the lists to numpy arrays
    X = np.array(X, dtype = float)
    y = np.array(y)
    return X, y

def encode_labels(labels):
    """
    We need to encode the labels to integers to let to softmax function work properly in the output layer
    """
    classes = sorted(np.unique(labels))    # e.g. ['Bad', 'Good', 'Medium'] because it is sorted alphabetically

    # creating to dictionaries, one for converting labels to integers, and one the other way around
    class_to_int = {cls: i for i, cls in enumerate(classes)} 
    int_to_class = {i: cls for cls, i in class_to_int.items()}

    # We need the integer values for backwardpropagation
    y_int = np.array([class_to_int[label] for label in labels], dtype=int)

    return y_int, class_to_int, int_to_class


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
    def __init__(self, X, y, hidden_size = (3,2), activation_function = "ReLU", learning_rate=0.01, epochs=1000):

        #determining the input and output sizes
        n_samples, n_features = X.shape

        #store the encoding of the y values
        self.y_int, self.class_to_int, self.int_to_class = encode_labels(y)

        #assigning all the values 
        self.n_samples = n_samples
        self.input_size = n_features
        self.hidden_size = hidden_size
        self.output_size = len(self.class_to_int)
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.epochs = epochs
        
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
        Here we use the output of the Architecture class to assign a weight/parameter to every connection between the nodes 
        """
        layer_sizes = self.architecture.layer_sizes
        self.weights = []
        self.biases = []

        # creating the weights and biases
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]

            #Chosing between initialization methods based on hidden layer activation function
            if self.activation_function.lower() == "relu":
                initialization_method = self.he_init(in_dim)
            elif self.activation_function.lower() in ["sigmoid", "tanh"]:
                initialization_method = self.xavier_init(in_dim)
            else:
                raise ValueError("Activation function must be 'ReLU' (standard), 'sigmoid' or 'tanh'.")

            W = np.random.randn(in_dim, out_dim)# * initialization_method #apply the initialization method to avoid vanishing or exploding weights
            b = np.zeros((1, out_dim))

            self.weights.append(W)
            self.biases.append(b)

    def forwardpropagation(self, X):
        """
        Perform a forward pass through the network.
        Returns:
            activations: list of activations per layer (including input layer)
            zs: list of pre-activation z-values per layer
        """
        activations = [X] # a0 = input layer
        z_values = [] # to store z-values

        a = X
        n_layers = len(self.weights)

        for i in range(n_layers):
            W = self.weights[i]
            b = self.biases[i]

            # compute pre-activation
            z = a @ W + b
            
            # store every z-value in it's list
            z_values.append(z)

            # output layer needs softmax
            if i == n_layers - 1:
                a = self.softmax(z)

            else:
                a= self.activation(z)
            
            # store every activation in the it's list
            activations.append(a)
        return activations, z_values

    def backwardpropagation(self, activations, z_values, learning_rate, y_int):
        """
        Calculates the updates for all the weights through every hidden layer
        """
        
        # One-hot encode
        y_onehot = self.one_hot(y_int)     # shape: (n_samples, output_size)

        # Output error
        output_error = activations[-1] - y_onehot

        # Gradient storage
        dWeights = [None] * len(self.weights)
        dbiases = [None] * len(self.biases)

        # Output layer gradients
        dWeights[-1] = activations[-2].T @ output_error / self.n_samples
        dbiases[-1] = np.sum(output_error, axis=0, keepdims=True) / self.n_samples

        #backwardpropagation through the hidden layers
        for layer in range(len(self.weights) -2, -1, -1):
            
            next_weight = self.weights[layer + 1] # weights of next layer

            output_error = (output_error @ next_weight.T) * self.activation_derivative(z_values[layer])

            # Gradients for this layer
            dWeights[layer] = activations[layer].T @ output_error / self.n_samples
            dbiases[layer] = np.sum(output_error, axis=0, keepdims=True) / self.n_samples

        for layer in range(len(self.weights)):
            self.weights[layer] -= learning_rate * dWeights[layer]
            self.biases[layer] -= learning_rate * dbiases[layer]

    def fit(self, X):
        """
        Trains the MLP using forward -> loss -> backward updates.
        """
        y_int = self.y_int

        for epoch in range(self.epochs):
            
            # 1. Forward pass
            activations, z_values = self.forwardpropagation(X)

            # 2. Compute loss
            y_onehot = self.one_hot(y_int)
            loss = self.cross_entropy(y_onehot, activations[-1])

            # 3. Accuracy
            preds = np.argmax(activations[-1], axis=1)
            acc = np.mean(preds == y_int)

            # 4. Backprop
            self.backwardpropagation(activations, z_values, self.learning_rate, y_int)

            # 5. Progress print
            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Loss = {loss:.4f} | Accuracy = {acc:.4f}")
        print("Training complete.")

    def predict(self, X):
        activations, _ = self.forwardpropagation(X)
        pred_int = np.argmax(activations[-1], axis=1)
        pred_labels = np.array([self.int_to_class[i] for i in pred_int])
        return pred_labels

    #=====================================================================================
    # Below are all my helper functions to shorten the fundamental calculation functions
    #=====================================================================================

    def one_hot(self, y_int):
        """
        Converts integer class labels into one-hot encoded format.
        
        Parameters:
            y_int (numpy.ndarray): Array of integer-encoded labels, shape (n_samples,)
        
        Returns:
            numpy.ndarray: One-hot encoded matrix of shape (n_samples, n_classes)
        """
        m = y_int.shape[0]
        onehot = np.zeros((m, self.output_size))
        onehot[np.arange(m), y_int] = 1
        return onehot
    
    def cross_entropy(self, y_onehot, y_pred):
        """
        Computes stable categorical cross-entropy loss.
        
        Parameters:
            y_onehot (numpy.ndarray): One-hot encoded ground truth labels (n_samples, n_classes)
            y_pred (numpy.ndarray): Predicted probabilities from softmax (n_samples, n_classes)
        
        Returns:
            float: mean cross-entropy loss
        """
        # to avoid log(0) we need to let values only get really close to it
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)  
        
        losses = -np.sum(y_onehot * np.log(y_pred), axis=1)
        return np.mean(losses)

        
    #activation function selector
    def activation(self, z):
        if self.activation_function == "ReLU":
            return self.relu(z)
        elif self.activation_function == "sigmoid":
            return self.sigmoid(z)
        elif self.activation_function == "tanh":
            return self.tanh(z)
        else:
            raise ValueError("Activation function must be 'ReLU' (standard), 'sigmoid' or 'tanh'.")
    
    #activation functions
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def relu(self, z):
        return np.maximum(0.0, z)
    
    def tanh(self, z):
        return np.tanh(z)
    
    #derivative activation function selector
    def activation_derivative(self, z):
        if self.activation_function == "ReLU":
            return self.relu_derivative(z)
        elif self.activation_function == "sigmoid":
            return self.sigmoid_derivative(z)
        elif self.activation_function == "tanh":
            return self.tanh_derivative(z)
        else:
            raise ValueError("Activation function must be 'ReLU' (standard), 'sigmoid' or 'tanh'.")
    
    #derivatives of activation functions
    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def tanh_derivative(self, z):
        return 1 - np.tanh(z)**2

    #softmax is used in the output layer to make a enable the model to classify with multiple classes
    def softmax(self, z): 
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    #Initialization methods to prevent vanishing or exploding weights
    def xavier_init(self, in_dim):
        return np.sqrt(1 / in_dim) #used for sigmoid or tanh networks
    
    def he_init(self, in_dim):
        return np.sqrt(2 / in_dim) #used for ReLU networks
        

X_train, y_train = data_extraction_csv("data\\train.csv")
X_test, y_test = data_extraction_csv("data\\test.csv")

def train_model(X_train, y_train, **hyperparams):
    mlp = MultiLayerPerceptron(X_train, y_train, **hyperparams)
    mlp.fit(X_train, **hyperparams)
    return {"model": mlp}

def predict(X_test, params):
    mlp = params["model"]
    return mlp.predict(X_test)
