import numpy as np

# The purpose of the model is to accurately categorise the sleep quality of each sample in the categories: bad, medium and good,
# based on the variables: Age, sleep duration, study hours, screen time, caffeine intake and physical activity.
# This means that we 6 variables for the input layer (+1 bias node), and 3 categories for the output layer.

def data_extraction_csv(csv_file):
    """ 
    This function extracts data from a csv file and splits the inputs and outputs in different arrays called X and y,
    assuming that the csv file has a header in the first row and the output variables stored in the last column 
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
    X = np.array(X)
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
    def __init__(self, X, y, hidden_size = (3,2), activation_function = "ReLU"):

        #determining the input and output sizes
        n_features = X.shape[1]
        outputs = np.unique(y)

        #assigning all the values 
        self.input_size = n_features
        self.hidden_size = hidden_size
        self.output_size = len(outputs)
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
        Here we use the output of the Architecture class to assign a weight/parameter to every connection between the nodes 
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
                raise ValueError("Activation functions supported: 'sigmoid', 'tanh', 'ReLU'")

            W = np.random.randn(in_dim, out_dim) * initialization_method #apply the initialization method to avoid vanishing or exploding weights
            b = np.zeros((1, out_dim))

            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        pass
    
    def backpropagation():
        pass

    def train_model(self, X_train, y_train, alpha = 0.1, learning_rate = 0.01, epochs = 1000, threshold_medium = 0.3, thershold_good = 0.8):
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

    def derivative():
        pass

    #activation functions
    def sigmoid(self, z): #optional activation function in hidden layer
     return 1 / (1 + np.exp(-z))
    def relu(self, z): #optional activation function in hidden layer
        return np.maximum(0.0, z)
    def softmax(self, z): #softmax is used in the output layer to make a enable the model to classify with multiple classes
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    #Initialization methods to prevent vanishing or exploding weights
    def xavier_init(self, in_dim):
        return np.sqrt(1 / in_dim) #used for sigmoid or tanh networks
    def he_init(self, in_dim):
        return np.sqrt(2 / in_dim) #used for ReLU networks
        


X, y = data_extraction_csv("data\\train.csv")

model = MultiLayerPerceptron(X, y)
