import torch
from NeuralNetwork.Evolving.Nodes import Nodes


class DynamicLayer:
    def __init__(self, out_dim):

        self.node_names = []
        self.out_dim = out_dim

        self.node = {}

    def fix_node_id(self):
        self.node_names = list(self.node.keys())

    def merge_nodes(self, node_list, merged_feature, merging_algo="avg"):
        print("Merge")
        merged_node = Nodes(merged_feature, self.out_dim)

        if merging_algo == "avg":
            node_1 = self.node[node_list[0]]
            node_2 = self.node[node_list[1]]

            weight_1, bias_1 = node_1.weight, node_1.bias
            weight_2, bias_2 = node_2.weight, node_2.bias
            with torch.no_grad:
                merged_node.weight = (weight_1 + weight_2)/2
                merged_node.bias = (bias_1 + bias_2)/2

        for node_name in node_list:
            del self.node[node_name]

        self.node[merged_feature] = merged_node
        self.fix_node_id()

    def check_add_node(self, names):
        self.fix_node_id()
        new_names = list(set(names) - set(self.node_names))
        for new_name in new_names:
            self.node[new_name] = Nodes(new_name, self.out_dim)
        self.fix_node_id()

    def build_input(self, inp):
        def build_dic_inp(inp_):
            temp = {}
            for point_, name_ in inp_:
                temp[name_] = point_
            return temp

        inp = build_dic_inp(inp)
        x = {}

        for name in self.node_names:
            if name in inp.keys():
                x[name] = (inp[name])
            else:
                x[name] = 0

        return x

    def forward(self, inp):
        names = [i[1] for i in inp]
        self.check_add_node(names)
        x = self.build_input(inp)

        out_list = []

        for name in self.node_names:
            node = self.node[name]
            point = torch.tensor(x[name])
            out_list.append(node.forward(point))

        out = out_list[0]
        for i in out_list[1:]:
            out = out + i

        return out

    def clear_grad(self):
        for node_name in self.node_names:
            self.node[node_name].clear_grad()

    def backward_propagation(self):
        for node_name in self.node_names:
            self.node[node_name].backward_propagation()
