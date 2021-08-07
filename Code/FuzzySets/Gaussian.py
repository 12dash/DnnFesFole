import math
import statistics
from FuzzySets.FuzzySet import FuzzySet


class Gaussian(FuzzySet):
    def __init__(self, id_, learning_rate=0.5, adapt_lr=True, **kwargs):
        super().__init__(id_, learning_rate, adapt_lr, **kwargs)
        self.stdev = 1
        self.lr = 0.1

    def update_membership(self):
        if len(self.points) >= 2:
            print(self.stdev)
            self.stdev = self.stdev - self.lr * statistics.stdev(self.points)

    def get_membership_value(self, point):
        try:
            gFunc = round((1/((2*math.pi)**0.5))*(1/self.stdev)*math.exp((-0.5)*((point-self.center)*2/(self.stdev**2))),4)
        except Exception as e:
            gFunc = 0
        return gFunc

    def update(self, point):
        self.points.append(point)
        self.update_center(point)
        self.update_membership()
