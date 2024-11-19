import numpy as np

class MLP:
    def __init__(self, n_inputs, n_hidden, n_outputs, learning_rate, activation='logistic'):
        self.w_hidden = np.random.randn(n_inputs, n_hidden) * 0.01
        self.b_hidden = np.zeros((1, n_hidden))
        self.w_output = np.random.randn(n_hidden, n_outputs) * 0.01
        self.b_output = np.zeros((1, n_outputs))
        self.learning_rate = learning_rate
        self.activation = activation

    def activation_function(self, x, derivative=False):
        if self.activation == 'linear':
            return 1 / 10 if derivative else x / 10
        elif self.activation == 'logistic':
            fx = 1 / (1 + np.exp(-x))
            return fx * (1 - fx) if derivative else fx
        elif self.activation == 'tanh':
            if derivative:
                tanh_x = np.tanh(x)
                return 1 - tanh_x ** 2
            return np.tanh(x)

    def forward(self, X):
        self.hidden_net = np.dot(X, self.w_hidden) + self.b_hidden
        self.hidden_output = self.activation_function(self.hidden_net)
        self.output_net = np.dot(self.hidden_output, self.w_output) + self.b_output
        self.output = self.activation_function(self.output_net)
        return self.output

    def backward(self, X, y, output):
        output_error = y - output
        output_delta = output_error * self.activation_function(self.output_net, derivative=True)
        hidden_error = np.dot(output_delta, self.w_output.T)
        hidden_delta = hidden_error * self.activation_function(self.hidden_net, derivative=True)
        self.w_output += self.learning_rate * np.dot(self.hidden_output.T, output_delta)
        self.b_output += self.learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        self.w_hidden += self.learning_rate * np.dot(X.T, hidden_delta)
        self.b_hidden += self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)