from FuzzySets.Discrete import Discrete
from FuzzySets.Gaussian import Gaussian
from Util.VisualizationHelper import save_plot


class Fuzzify:
    def __init__(self, name, c_type="Discrete"):
        self.name = name
        self.cType = c_type

        self.fSets = []
        self.threshold = 0.4
        self.merged_nodes = []

        self.id_ = 0

    def new_set(self):
        id_ = self.name + "_" + str(self.id_)
        fSet = Gaussian(id_) if self.cType == "Gaussian" else Discrete(id_)
        self.id_ += 1
        return fSet

    def get_membership_val(self, point, with_id=True):
        temp = []
        for fSet in self.fSets:
            if with_id:
                temp.append([fSet.get_membership_value(point), fSet.id])
            else:
                temp.append(fSet.get_membership_value(point))
        return temp

    def create_add_point(self, point):
        fSet = self.new_set()
        fSet.update(point)
        self.fSets.append(fSet)

    def get_common_area(self, f_set_a, f_set_b):
        common = set(f_set_a.membershipValue.keys()).intersection(set(f_set_b.membershipValue.keys()))
        common_area = 0
        if len(common) > 0:
            for i in list(common):
                common_area += min(f_set_a.membershipValue[i], f_set_b.membershipValue[i])
            max_area = max(common_area/f_set_a.get_area(), common_area/f_set_b.get_area())
            return max_area
        return 0

    def merge(self, set_a, set_b):
        new_f_set = self.new_set()
        points = set_a.points + set_b.points
        points.sort()
        for point in points:
            new_f_set.update(point)
        self.merged_nodes = [[set_a.id, set_b.id], new_f_set]
        self.fSets.remove(set_a)
        self.fSets.remove(set_b)
        self.fSets.append(new_f_set)

    def check_merge(self):
        for i in range(len(self.fSets)-1):
            max_overlap = []
            for j in range(i+1, len(self.fSets)):
                overlap = self.get_common_area(self.fSets[i], self.fSets[j])
                if 0.7 < overlap <= 1:
                    if len(max_overlap) > 0:
                        if max_overlap[0] < overlap:
                            max_overlap = [overlap, self.fSets[j]]
                    else:
                        max_overlap = [overlap, self.fSets[j]]
            if len(max_overlap) > 0:
                self.merge(self.fSets[i], max_overlap[1])

    def add_point(self, point):
        self.check_merge()
        if len(self.fSets) == 0:
            self.create_add_point(point)
        else:
            candidate = None
            for fSet in self.fSets:
                membershipValue = fSet.get_membership_value(point)
                if membershipValue > self.threshold:
                    if candidate is None:
                        candidate = fSet
                    elif membershipValue > candidate.get_membership_value(point):
                        candidate = fSet

            if candidate is None:
                self.create_add_point(point)
            else:
                candidate.update(point)

    def print_sets(self):
        print(len(self.fSets))
        for i in self.fSets:
            i.details_()

    def plot_membership(self, dataset):
        path = f"./Runs/{dataset}/{self.name}.png"
        temp = []
        for fSet in self.fSets:
            temp.append(fSet.membershipValue)
        save_plot(temp, self.name, path)
        return

