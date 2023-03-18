import torch


class Nodes:
    def __init__(self, name, out_dim, learning_rate=0.01):

        self.name = name
        self.out_dim = out_dim

        self.learning_rate = learning_rate

        self.weight = None
        self.bias = None

        self.initialize_params()

    def initialize_params(self):
        """
        Initialize the weight and biases of the nodes
        """

        self.weight = torch.randn(self.out_dim, requires_grad=True)
        self.bias = torch.randn(self.out_dim, requires_grad=True)

    def clear_grad(self):
        """
        Clear the gradient of the weight and bias
        :return: Nothing
        """
        self.weight.grad.zero_()
        self.bias.grad.zero_()

    def backward_propagation(self):
        """
        Backward Propagation of the gradients.
        :return: Nothing
        """
        with torch.no_grad():
            self.weight -= self.learning_rate*self.weight.grad
            self.bias -= self.learning_rate*self.bias.grad

        self.clear_grad()

    def forward(self, x):
        """
        Forward propagation of the node

        :param x: point for the forward propagation
        :return: The output of the node layers
        """
        return self.weight * x + self.bias

