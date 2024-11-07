# key_lattice.py

"""
KeyLattice class representing the key lattice structure.
"""

import threading
import hashlib

import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting toolkit
import numpy as np
# from sklearn.decomposition import PCA

class KeyLattice:
    """
    Represents the key lattice structure for a participant.
    Manages keys, updates, and visualization.
    """

    def __init__(self, initial_key, dimension):
        self.dimension = dimension
        self.lattice = {}  # {(index_tuple): key}
        self.lattice_lock = threading.Lock()
        initial_index = tuple([0] * self.dimension)
        self.lattice[initial_index] = initial_key
        self.max_index = initial_index

        # For visualization
        self.graph = nx.DiGraph()
        self.graph.add_node(initial_index, key=initial_key)

    def get_max_key(self):
        """
        Get the current maximal key and its index.
        """
        with self.lattice_lock:
            return self.lattice[self.max_index], self.max_index

    def add_key(self, index, key, predecessor_index=None, x=None):
        """
        Add a new key to the lattice and update the graph.
        """
        self.lattice[index] = key
        self.update_max_index(index)
        # Update the graph for visualization
        self.graph.add_node(index, key=key)
        if predecessor_index:
            self.graph.add_edge(predecessor_index, index, x=x)

    def update_max_index(self, index):
        """
        Update the maximal index of the lattice.
        """
        self.max_index = tuple(max(i, j) for i, j in zip(self.max_index, index))

    def print_lattice(self):
        """
        Print the current state of the lattice.
        """
        print(f"Lattice state for Participant {self.dimension}:")
        for index in sorted(self.lattice.keys()):
            key = self.lattice[index]
            key_repr = hashlib.sha256(key).hexdigest()[:8]
            print(f"  Index {index}: Key hash {key_repr}")
        print()

    def visualize_lattice(self, participant_index, cycle):
        """
        Visualize the key lattice in 3D.
        """

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Extract positions from lattice indices
        nodes = list(self.graph.nodes())
        indices = np.array(nodes)
        # Handle cases where dimension > 3
        if self.dimension > 3:
            pca = PCA(n_components=3)
            indices_3d = pca.fit_transform(indices)
        else:
            indices_3d = indices

        pos = {node: indices_3d[i] for i, node in enumerate(nodes)}
        xs = indices_3d[:, 0]
        ys = indices_3d[:, 1]
        zs = indices_3d[:, 2]
        # Draw nodes
        ax.scatter(xs, ys, zs, s=100, c='lightblue', depthshade=True)
        # Annotate nodes with their key values
        for node in self.graph.nodes():
            key = self.lattice[node]
            key_repr = hashlib.sha256(key).hexdigest()[:8]
            x, y, z = pos[node]
            ax.text(x, y, z, f"{node}\n{key_repr}", fontsize=9, ha='center')
        # Draw edges and annotate with x values
        for edge in self.graph.edges(data=True):
            src, dst, data = edge
            x_values = [pos[src][0], pos[dst][0]]
            y_values = [pos[src][1], pos[dst][1]]
            z_values = [pos[src][2], pos[dst][2]]
            ax.plot(x_values, y_values, z_values, c='gray')
            # Calculate the midpoint for labeling
            mid_x = (pos[src][0] + pos[dst][0]) / 2
            mid_y = (pos[src][1] + pos[dst][1]) / 2
            mid_z = (pos[src][2] + pos[dst][2]) / 2
            # Get the x value (random data)
            x_value = data.get('x')
            if x_value:
                x_repr = hashlib.sha256(x_value).hexdigest()[:8]
                ax.text(mid_x, mid_y, mid_z, f"x: {x_repr}", fontsize=8, color='red', ha='center')
        ax.set_xlabel('Dimension 0')
        ax.set_ylabel('Dimension 1')
        ax.set_zlabel('Dimension 2')
        ax.set_title(f"Lattice Visualization for Participant {participant_index} at Cycle {cycle}")
        plt.show()

    def forget(self, window_index):
        """
        Forget keys outside the window defined by window_index.
        Remove corresponding nodes from the graph.
        """
        keys_to_delete = []
        for index in list(self.lattice.keys()):
            if all(i <= w for i, w in zip(index, window_index)):
                keys_to_delete.append(index)
        for index in keys_to_delete:
            del self.lattice[index]
            # Remove node from graph
            if index in self.graph:
                self.graph.remove_node(index)
