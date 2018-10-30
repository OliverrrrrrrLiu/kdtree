import numpy as np
import heapq


class KdNode(object):
    def __init__(self, data, left = None, right = None, axis = None,
                 statistics = None, bbox = None, index = None):
        self.data = data
        self.left = left
        self.right = right
        self.axis = axis
        self.statistics = statistics
        self.bbox = bbox
        self.index = index

    def is_leaf(self):
        return self is not None and self.left is None and self.right is None

    def is_contained(self, a, b):
        return (a <= self.bbox['lb']).all() and (b >= self.bbox['ub']).all()

    def is_disjoint(self, a, b):
        return (a >= self.bbox['ub']).all() or (b <= self.bbox['lb']).all()

    def dist_2_bbox(self, p):
        dimension = self.bbox['lb'].shape[0]
        vector_2_bbox = np.maximum(np.zeros(dimension), np.maximum(self.bbox['lb'] - p, p - self.bbox['ub']))
        return np.sum(vector_2_bbox ** 2) ** 0.5

class KdTree():
    def __init__(self):
        self.tree = None

    def build(self, data, leaf_size):
        root_node = KdNode(data)
        dimension, n_sample = data.shape
        root_node.index = np.array(range(n_sample))
        self.tree = self.init_KdTree(root_node, leaf_size)

    def init_KdTree(self, KdTree, leaf_size):
        data, index = KdTree.data, KdTree.index
        dimension, n_sample = data.shape

        if n_sample <= leaf_size:
            statistics = {}
            statistics['mean'] = np.sum(data, axis = 1) / (n_sample * 1.0)
            statistics['variance'] = np.sum((data - statistics['mean'].reshape((dimension,1))) ** 2, axis = 1) / (n_sample * 1.0)
            KdTree.statistics = statistics
            KdTree.bbox = self.bounding_box(data)
            return KdTree

        max_spread_axis = np.argmax(np.amax(data, axis = 1) - np.amin(data, axis = 1))
        sorted_arg = np.argsort(data[max_spread_axis, :])
        data, index = data[:, sorted_arg], index[sorted_arg]

        KdTree.data = data[max_spread_axis, n_sample/2]
        KdTree.axis = max_spread_axis
        KdTree.bbox = self.bounding_box(data)

        left_node, right_node = KdNode(data[:, :n_sample/2]), KdNode(data[:, n_sample/2:])
        left_node.index, right_node.index = index[:n_sample/2], index[n_sample/2:]
        KdTree.left = self.init_KdTree(left_node, leaf_size)
        KdTree.right = self.init_KdTree(right_node, leaf_size)
        return KdTree

    def bounding_box(self, data):
        dimension, n_sample = data.shape
        bbox = {}
        bbox['lb'] = np.amin(data, axis = 1)
        bbox['ub'] = np.amax(data, axis = 1)
        return bbox

    def traverse(self, KdTree, accum, pre_visit_fun = None, visit_fun = None, post_visit_fun = None):
        if pre_visit_fun is not None:
            accum = pre_visit_fun(KdTree, accum)
        if not KdTree.is_leaf():
            accum = self.traverse(accum, KdTree.left, pre_visit_fun, visit_fun, post_visit_fun)

            if visit_fun is not None:
                accum = visit_fun(start, accum)

            accum = self.traverse(accum, KdTree.right, pre_visit_fun, visit_fun, post_visit_fun)
        if post_visit_fun is not None:
            accum = post_visit_fun(start, accum)
        return accum


    def inRange(self, KdTree, a, b):
        if KdTree is None or KdTree.is_disjoint(a, b):
            return
        elif KdTree.is_leaf():
            data = KdTree.data
            dimension, n_sample = data.shape
            check_lb = (a.reshape((dimension, 1)) <= data).all(axis = 0)
            check_ub = (data <= b.reshape((dimension, 1))).all(axis = 0)
            inRange_idx = np.where(np.logical_and(check_lb, check_ub))[0]
            return data[:, inRange_idx]

        inRange_data_left, inRange_data_right = self.inRange(KdTree.left, a, b), self.inRange(KdTree.right, a, b)
        if inRange_data_left is not None and inRange_data_right is not None:
            return np.hstack((inRange_data_left, inRange_data_right))
        elif inRange_data_left is not None:
            return inRange_data_left
        elif inRange_data_right is not None:
            return inRange_data_right
        return

    def distance_query(self, KdTree, p, r):
        if KdTree is None or KdTree.dist_2_bbox(p) > r:
            return
        elif KdTree.is_leaf():
            data = KdTree.data
            dimension, n_sample = data.shape
            dist = np.sum((data - p.reshape((dimension, 1))) ** 2, axis = 0) ** 0.5
            return data[:, np.where(dist <= r)[0]]

        inDist_data_left, inDist_data_right = self.distance_query(KdTree.left, p, r), self.distance_query(KdTree.right, p, r)
        if inDist_data_left is not None and inDist_data_right is not None:
            return np.hstack((inDist_data_left, inDist_data_right))
        elif inDist_data_left is not None:
            return inDist_data_left
        elif inDist_data_right is not None:
            return inDist_data_right
        return

    def knn_query(self, KdTree, p, k, r_max = np.inf):
        knn = []
        for i in range(k):
            heapq.heappush(knn, (np.inf, None))
        return [item[1] for item in list(heapq.nsmallest(k, self.find_knn(KdTree, p, k, knn, np.inf)))]

    def find_knn(self, KdTree, p, k, cur_knn, cur_max, r_max = np.inf):
        if KdTree is None or KdTree.dist_2_bbox(p) > cur_max:
            print('here')
            return cur_knn
        elif KdTree.is_leaf():
            data, index = KdTree.data, KdTree.index
            dimension, n_sample = data.shape
            dist = np.sum((data - p.reshape((dimension, 1))) ** 2, axis = 0) ** 0.5
            for i in range(n_sample):
                heapq.heappush(cur_knn, (dist[i], {'data': data[:,i], 'index': index[i]}))
            return cur_knn

        cur_max = heapq.nsmallest(k, cur_knn)[k-1][0]
        cur_knn = self.find_knn(KdTree.left, p, k, cur_knn, cur_max)
        cur_max = heapq.nsmallest(k, cur_knn)[k-1][0]
        cur_knn = self.find_knn(KdTree.right, p, k, cur_knn, cur_max)
        return cur_knn

    def max_distance_rectangle(self, bbox1, bbox2):
        return np.sum(np.maximum(bbox1['ub'] - bbox2['lb'], bbox2['ub'] - bbox1['lb']) ** 2) ** 0.5

    def min_distance_rectangle(self, bbox1, bbox2):
        return np.sum(np.maximum(0, np.maximum(bbox1['lb'] - bbox2['ub'], bbox2['lb'] - bbox1['ub'])) ** 2) ** 0.5

    def pairs_query(self, KdTree1, KdTree2, r):
        n = KdTree1.index.shape[0]
        results = [[] for i in range(n)]
        def traverse_checking(node1, node2):
            if self.min_distance_rectangle(node1.bbox, node2.bbox) > r:
                return
            elif self.max_distance_rectangle(node1.bbox, node2.bbox) < r:
                traverse_no_checking(node1, node2)
            elif node1.is_leaf():
                if node2.is_leaf():
                    dimension, n_sample = node1.data.shape
                    for i in range(n_sample):
                        p = node1.data[:,i]
                        dist = np.sum((node2.data - p.reshape((dimension, 1))) ** 2, axis = 0) ** 0.5
                        results[node1.index[i]] += node2.index[np.where(dist <= r)[0]].tolist()
                else:
                    traverse_checking(node1, node2.left)
                    traverse_checking(node1, node2.right)
            elif node2.is_leaf():
                traverse_checking(node1.left, node2)
                traverse_checking(node1.right, node2)
            else:
                traverse_checking(node1.left, node2.left)
                traverse_checking(node1.left, node2.right)
                traverse_checking(node1.right, node2.left)
                traverse_checking(node1.right, node2.right)

        def traverse_no_checking(node1, node2):
            if node1.is_leaf():
                if node2.is_leaf():
                    for i in range(node1.index.shape[0]):
                        p = node1.data[:,i]
                        results[node1.index[i]] += node2.index.tolist()
                else:
                    traverse_no_checking(node1, node2.left)
                    traverse_no_checking(node1, node2.right)
            else:
                traverse_no_checking(node1.left, node2)
                traverse_no_checking(node1.right, node2)

        traverse_checking(KdTree1, KdTree2)
        return results





















a = np.array([[0,1,19],[5,11,2],[24,2,40]])
b = np.random.randint(10, size = 70).reshape((7,10))
c = np.array([[0,1,5.5,7,10,9],[1,1,6,3,8.5,4],[4,7,3,6,7,5],[8,4,3,1,2,6]])
d = np.array([19.7, 21.4, 25.6, 29.4, 30.1, 30.1, 30.4, 30.7, 31.2, 32.8, 33, 33.3, 38.1, 43.3, 43.5, 47.9]).reshape((1,16))
test = KdTree()
test.build(d, 2)
tree = test.tree

ir = test.inRange(tree, np.array([27.0]), np.array([35.0]))
knn = test.knn_query(tree, np.array([31.0]), 5)
dist_query = test.distance_query(tree, np.array([30.0]), 10)
result = test.pairs_query(tree, tree, 29)

def print_tree(tree):
    if tree.is_leaf():
        print(tree.bbox)
        print(tree.statistics)
        print(tree.data)
    else:
        #print(tree.axis)
        print_tree(tree.left)
        print_tree(tree.right)
#print_tree(tree)
