import numpy as np
from utils import rmse, grad_mse, hess_mse

class Node:
    def __init__(self):
        self.threshold = 0
        self.k = 0
        self.leaf = True
        self.left = None # ノードオブジェクト（左）
        self.right = None # ノードオブジェクト（右）
        self.w = 0
    def divide(self, x):
        if x[self.k] >= self.threshold: # １サンプル
            return self.left
        else:
            return self.right

class DecisionTree:
    def __init__(self, learning_rate=1):
        self.root_node = None
        self.next_tree = None
        self.lr = learning_rate
    def __call__(self, x):
        n = self.root_node
        while not n.leaf:
            n = n.divide(x)
        return n.w * self.lr

class XGBDTRegressor:
    def __init__(self, max_trees=1000,r=0., l2=0., learning_rate=0.1):
        """
        r: [0,1], number of nodes regularization paremeter
        l2: [0,1], l2 regularization parameter
        """
        self.root_tree = DecisionTree()
        self.loss_fn = rmse
        self.r = r
        self.l2 = l2
        self.gradient = grad_mse
        self.hessian = hess_mse
        self.J = max_trees # 木の最大数
        self.max_depth = 3 # 個々の木の最大深さ
        self.learning_rate = learning_rate
    def fit(self, X, y):
        N, M = X.shape
        self.base_score = np.mean(y)
        # ダミーの先頭
        self.root_tree = DecisionTree()
        prev_tree = self.root_tree

        for j in range(self.J):
            # 今回追加する新しい木
            new_tree = DecisionTree(learning_rate=self.learning_rate)
            new_tree.root_node = Node()

            # これまでの木の予測
            y_pred = self.predict(X)

            # 勾配とヘシアン
            g = self.gradient(y, y_pred)
            h = self.hessian(y, y_pred)

            # 各サンプルがどのノードにいるか
            node_assignments = np.ones(N, dtype=int)

            # 木全体で共有する辞書
            nodes = {1: new_tree.root_node}

            for depth in range(self.max_depth):
                for t in range(2**depth, 2**(depth + 1)):
                    if t not in nodes:
                        continue

                    N_t = (node_assignments == t)
                    if not np.any(N_t):
                        continue

                    current_node = nodes[t]

                    if depth == self.max_depth - 1:
                        current_node.w = -np.sum(g[N_t]) / (np.sum(h[N_t]) + self.l2)
                        continue

                    k, threshold, N_L, N_R = self._exact_greedy_algorithm(
                        X[N_t], g[N_t], h[N_t]
                    )

                    if k is None:
                        current_node.w = -np.sum(g[N_t]) / (np.sum(h[N_t]) + self.l2)
                        current_node.leaf = True
                        continue

                    current_node.leaf = False
                    current_node.left = Node()
                    current_node.right = Node()
                    current_node.k = k
                    current_node.threshold = threshold

                    nodes[2*t] = current_node.left
                    nodes[2*t+1] = current_node.right

                    global_idx = np.where(N_t)[0]
                    left_global = global_idx[N_L]
                    right_global = global_idx[N_R]

                    node_assignments[left_global] = 2 * t
                    node_assignments[right_global] = 2 * t + 1

            # 学習済み木を連結
            prev_tree.next_tree = new_tree
            prev_tree = new_tree
            print(f"iter {j+1}, loss = {self.loss(X, y)}")

    def predict(self, x):
        if x.ndim == 1:
            pred = self.base_score
            tree = self.root_tree.next_tree
            while tree is not None:
                pred += tree(x)
                tree = tree.next_tree
            return pred
        else:
            pred = np.full(x.shape[0], self.base_score)
            tree = self.root_tree.next_tree
            while tree is not None:
                pred += np.array([tree(xi) for xi in x])
                tree = tree.next_tree
            return pred
    def loss(self, X, y):
        return self.loss_fn(y, self.predict(X))
    def _exact_greedy_algorithm(self, X, g_t, h_t):
        N, M = X.shape
        G = np.sum(g_t)
        H = np.sum(h_t)

        best_gain = -np.inf
        best_k = None
        best_threshold = None

        for k in range(M):
            sort_idx = np.argsort(X[:, k])
            X_k = X[sort_idx, k]
            g_sorted = g_t[sort_idx]
            h_sorted = h_t[sort_idx]

            G_L_all = np.cumsum(g_sorted)[:-1]
            H_L_all = np.cumsum(h_sorted)[:-1]
            G_R_all = G - G_L_all
            H_R_all = H - H_L_all

            valid = X_k[:-1] != X_k[1:]
            if not np.any(valid):
                continue

            gains = 0.5 * (
                G_L_all**2 / (H_L_all + self.l2)
                + G_R_all**2 / (H_R_all + self.l2)
                - G**2 / (H + self.l2)
            ) - self.r

            gains[~valid] = -np.inf
            idx = np.argmax(gains)

            if gains[idx] > best_gain:
                best_gain = gains[idx]
                best_k = k
                best_threshold = 0.5 * (X_k[idx] + X_k[idx + 1])

        if best_k is None or best_gain <= 0:
            return None, None, None, None

        left_mask = X[:, best_k] >= best_threshold
        right_mask = ~left_mask
        N_L = np.where(left_mask)[0]
        N_R = np.where(right_mask)[0]
        return best_k, best_threshold, N_L, N_R