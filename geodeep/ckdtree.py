# Originally based from https://github.com/PUTvision/qgis-plugin-deepness

import heapq

import numpy as np


class cKDTree:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.tree = self._build_kdtree(np.arange(len(data)))

    def _build_kdtree(self, indices, depth=0):
        if len(indices) == 0:
            return None
        axis = depth % self.data.shape[1]  # alternate between x and y dimensions
        
        sorted_indices = indices[np.argsort(self.data[indices, axis])]
        mid = len(sorted_indices) // 2
        node = {
            'index': sorted_indices[mid],
            'left': self._build_kdtree(sorted_indices[:mid], depth + 1),
            'right': self._build_kdtree(sorted_indices[mid + 1:], depth + 1)
        }
        return node

    def query(self, point: np.ndarray, k: int):
        if type(point) is not np.ndarray:
            point = np.array(point)
        
        return [index for _, index in self._query(point, k, self.tree)]
    
    def _query(self, point, k, node, depth=0, best_indices=None, best_distances=None):
        if node is None:
            return None
        
        axis = depth % self.data.shape[1]
        
        if point[axis] < self.data[node['index']][axis]:
            next_node = node['left']
            other_node = node['right']
        else:
            next_node = node['right']
            other_node = node['left']
        
        if best_indices is None:
            best_indices = []
            best_distances = []
        
        current_distance = np.linalg.norm(self.data[node['index']] - point)
        
        if len(best_indices) < k:
            heapq.heappush(best_indices, (-current_distance, node['index']))
        elif current_distance < -best_indices[0][0]:
            heapq.heappop(best_indices)
            heapq.heappush(best_indices, (-current_distance, node['index']))
        
        if point[axis] < self.data[node['index']][axis] or len(best_indices) < k or abs(point[axis] - self.data[node['index']][axis]) < -best_indices[0][0]:
            self._query(point, k, next_node, depth + 1, best_indices, best_distances)
        
        if point[axis] >= self.data[node['index']][axis] or len(best_indices) < k or abs(point[axis] - self.data[node['index']][axis]) < -best_indices[0][0]:
            self._query(point, k, other_node, depth + 1, best_indices, best_distances)
            
        return best_indices
