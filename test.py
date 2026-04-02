from sklearn.datasets import fetch_california_housing
from model import XGBDTRegressor, Node, DecisionTree
import numpy as np

class TestTree:
    def __init__(self):
        n1 = Node()
        n2 = Node()
        n3 = Node()
        n4 = Node()
        n5 = Node()
        n6 = Node()
        n7 = Node()
        self.tree = DecisionTree()
        self.tree.root_node = n1
        self.nodes = [n1, n2, n3, n4, n5, n6, n7]
    def run(self):
        x = np.array([0.4, 0.3, 0.7, 0.8, 0.2, 0.9, 0.2]) # dim=7
        thresholds = [0.5, 0.1, 0.4, 0.6, 0.9, 0.1, 0.3]
        for i, n in enumerate(self.nodes):
            n.threshold = thresholds[i]
            n.k = i
            if i < 3:
                n.leaf = False
                n.left = self.nodes[2*i+1]
                n.right = self.nodes[2*(i+1)]
            else:
                n.w = i
        if self.tree(x) == 5:
            print("ok")
class TestXGBPredictMethod:
    def __init__(self):
        self.nodes_1 = [Node() for _ in range(7)]
        self.nodes_2 = [Node() for _ in range(7)]
        self.tree_1 = DecisionTree()
        self.tree_2 = DecisionTree()
        self.tree_1.next_tree = self.tree_2
        self.tree_1.root_node = self.nodes_1[0]
        self.tree_2.root_node = self.nodes_2[0]
        self.model = XGBDTRegressor()
        self.model.root_tree = self.tree_1

    def run(self):
        x = np.array([0.4, 0.3, 0.7, 0.8, 0.2, 0.9, 0.2]) # dim=7
        thresholds_1 = [0.5, 0.1, 0.4, None, None, None, None]
        thresholds_2 = [0.1, 0.8, 0.6, None, None, None, None]
        w_1 = [None, None, None, 1, 2, 3, 4]
        w_2 = [None, None, None, 1, 2, 3, 4]
        self.init_nodes(self.nodes_1, thresholds_1, w_1)
        self.init_nodes(self.nodes_2, thresholds_2, w_2)
        if self.tree_1(x) + self.tree_2(x) == self.model.predict(x):
            print("ok")

    def init_nodes(self, nodes, thresholds, w):
        for i, n in enumerate(nodes):
            n.threshold = thresholds[i]
            n.k = i
            if i < 3:
                n.leaf = False
                n.left = nodes[2*i+1]
                n.right = nodes[2*(i+1)]
            else:
                n.w = w[i]
if __name__ == "__main__":
    california_housing_data = fetch_california_housing()

    X = california_housing_data.data
    y = california_housing_data.target
    col_name = california_housing_data.feature_names
    descr = california_housing_data.DESCR
    print(X.shape)
    print(y.shape)

    test_tree = TestTree()
    test_XGBpredictor = TestXGBPredictMethod()
    test_tree.run()
    # test_XGBpredictor.run()
    model = XGBDTRegressor(max_trees=100)
    print(np.mean(y))
    model.fit(X, y)

    