import math


class FuzzySet:
    def __init__(self, id_, learning_rate=0.5, adapt_lr=True, **kwargs):

        self.id = id_

        self.learningRate = learning_rate
        self.adaptLR = adapt_lr
        self.decayRate = 0.05

        self.fetch_parameters(kwargs)

        self.center = None
        self.points = []

    def fetch_parameters(self, kwargs):
        self.decayRate = kwargs.get("decay_rate", 0.01)

    def adapt_learning_rate(self):
        if self.adaptLR:
            self.learningRate = self.learningRate*math.exp(-1*self.decayRate)
            self.adaptLR = False if self.learningRate <= 0.01 else True

    def update_center(self, point):
        if self.center is None:
            self.center = point
        else:
            self.center = self.center - self.learningRate*(self.center-point)

    def details_(self):
        print("Center : ", self.center)
        print(sorted(self.points))
        print("_"*40)
