import math
from FuzzySets.FuzzySet import FuzzySet


class Discrete(FuzzySet):
    def __init__(self, id_, learning_rate=0.5, adapt_lr=True, **kwargs):
        super().__init__(id_, learning_rate, adapt_lr, **kwargs)

        self.membershipValue = {}
        self.membershipDecayRate = None
        self.width = None
        self.roundPlace = None

        self.fetch_parameters(kwargs)
        self.maxVal = None
        self.area = 0

    def fetch_parameters(self, kwargs):
        self.membershipDecayRate = kwargs.get("membership_decay_rate", 0.5)
        #self.width = kwargs.get("width", 0.01)

    def fetch_nearest_point(self, point):
        nearestPlace = round((point // self.width) * self.width, self.roundPlace)
        return nearestPlace

    def normalize_membership(self):
        maxVal = max(self.membershipValue.values())
        self.maxVal = maxVal
        for key in self.membershipValue:
            self.membershipValue[key] /= self.maxVal

    def un_normalize_membership(self):
        if self.maxVal is not None:
            for key in self.membershipValue:
                self.membershipValue[key] *= self.maxVal

    def update_membership_width(self, x):
        def estimate_width(x):
            x_str = str(x)
            if '.' in x_str:
                decimal = x_str.split(".")[1]

                self.roundPlace = len(str(decimal))
                self.roundPlace = self.roundPlace if self.roundPlace <= 2 else 2
                self.width = (10 ** (-1 * self.roundPlace))
                self.membershipDecayRate = 1
            else:
                self.width = 1
                self.roundPlace = 0

        if self.width is None:
            estimate_width(x)

    def get_membership_value(self, point):
        nearestPoint = self.fetch_nearest_point(point)
        if nearestPoint in self.membershipValue:
            return self.membershipValue[nearestPoint]
        else:
            return 0

    def check_modify_membership(self, update_point, update_value):
        if update_point in self.membershipValue:
            self.membershipValue[update_point] += update_value
        else:
            self.membershipValue[update_point] = update_value

    def update_membership_value(self, point):
        nearestPoint = self.fetch_nearest_point(point)
        self.check_modify_membership(nearestPoint, 1)

        dist = 1
        while dist < 1000:
            updatePointL = round(nearestPoint - dist * self.width, self.roundPlace)
            updatePointR = round(nearestPoint + dist * self.width, self.roundPlace)

            updateValue = (1-dist*0.1*self.membershipDecayRate) if self.width > 0.1 else (1-dist*self.width*self.membershipDecayRate)

            self.check_modify_membership(updatePointL, updateValue)
            self.check_modify_membership(updatePointR, updateValue)
            if updateValue <= 0:
                break
            dist += 1

    def update(self, point):
        self.un_normalize_membership()
        self.update_membership_width(point)
        self.points.append(point)
        self.update_center(point)
        self.update_membership_value(point)
        self.adapt_learning_rate()
        self.normalize_membership()
        self.get_area()

    def get_area(self):
        self.area = 0
        for i in self.membershipValue:
            self.area += self.membershipValue[i]
        return self.area

    def details_(self):
        super().details_()
