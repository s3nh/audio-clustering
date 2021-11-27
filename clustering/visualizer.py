import pyclustering
from pyclustering.cluster import cluster_visualizer
import matplotlib.pyplot as plt
import random
import numpy as np
from utils.utils import initialize_colors
from typing import List, Dict, Union, Tuple


def sklearn_results(alg, X: np.array) -> None:
    """Draft version of sklearn results visualisation.

    Parameters
    -----------
    object
        Pretrained algorithm
    lim : Tuple
        limes for created plot

    Returns
    --------
    None
    """
    labels = alg.labels_
    n_clusters = np.unique(labels).size
    _colors = initialize_colors(n_clusters)
    centroids = alg.subcluster_centers_
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.1, top=0.9)
    for centroid, k, col in zip(centroids, range(n_clusters), _colors):
        color = random.choice(_colors)
        mask = labels == k
        plt.scatter(X[mask, 0], X[mask, 1], color=color, marker=".", alpha=0.5)
    if n_clusters is None:
        plt.scatter(centroid[0], centroid[1], marker="+", c=color, s=25)
    plt.show()



def clustering_show_results(clusters: List, sample: List) -> None:
    """Pyclustering visualisation snippet.
    Parameters
    -----------
    clusters: List
        Within clusters observartion, shaped (n_clusters, n_obs)
    sample: List
        Visualized dataset

    Returns
    --------
    None
    """
    visualizer = cluster_visualizer()
    visualizer.append_clusters(clusters, sample)
    visualizer.show()