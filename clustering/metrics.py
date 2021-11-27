from abc import ABCMeta, abstractmethod
from clustering.clustering import PcaDataLoader
import numpy as np
import pickle
import sklearn
from sklearn.metrics import pairwise_distances, davies_bouldin_score
from typing import Dict, List, Union, Optional, TypeVar

T = TypeVar("T")


class MetricsAbstract(object):
    def __init__(self):
        self.storage: Dict = {}

    def set_storage(self):
        pass

    def get_storage(self):
        return self.storage

    def save_storage(self, path: str):
        pass

    def get_metric(self):
        pass

    def get_results(self):
        pass

class ClusteringMetrics(MetricsAbstract):
    """Reader for unsupervised algorithms metrics.
    Parameters
    -----------
    None

    Returns
    --------
    None
    """
    def __init__(self):
        super(MetricsAbstract, self).__init__()
        self.data = PcaDataLoader().data
        self.resultpath: str = None
        self.storage: Dict = dict()
        self.metrics: List = [
            "calinski_harabasz_score",
            "davies_bouldin_score",
            "silhouette_score",
            "homogeneity_score"
        ]

    def __call__(self, path: str):
        """
        Parameters
        -----------
        path : str
            path for processed .pkl saved file

        Returns
        --------
        None
        """
        alg = self.get_alg(path)
        labels = alg.labels_
        _tmpres = self.get_triple(labels)
        self.set_storage(path, _tmpres)

    def get_metric(self, metricname: str) -> Optional:
        """Get metric name.

        Parameters
        -----------
        metricname: str
            Name of metric which want to initialize

        Returns
        --------
        _metric: object
            Initialized object
        """
        assert metricname in sklearn.metrics.__all__, "Metric is not properly named"
        _metric = getattr(sklearn.metrics, metricname)
        return _metric

    def get_alg(self, path: str) -> T:
        """Load pickled object

        Parameters
        ----------
        path : str
            Path to .pkl file

        Returns
        ---------
        T
            Loaded object (fitted unsupervised algorithm)
        """
        with open(path, "rb") as algfile:
            _alg = pickle.load(algfile)
        return _alg

    def set_storage(self, path: str, _tmp: Dict) -> None:
        """Update storage with new keys, which create nested dictionary
        #TODO Rethink if thats necessary to store nested dictionary
        """
        self.storage[path] = _tmp

    def save_storage(self, path: str):
        """Self existed storage in .pkl file format.

        Parameters
        ----------
        None

        Returns
        --------
        None
        """
        assert len(self.storage.keys()) > 0, "Are you sure that storage should be empty?"
        with open(path, "wb") as storfile:
            pickle.dump(self.storage, storfile)

    def get_triple(self, labels: np.array) -> Dict:
        _tmp = dict()
        for metricname in self.metrics:
            _metric = self.get_metric(metricname)
            _tmp[metricname] = self.get_results_summary(labels, _metric)
        return _tmp

    def get_results_summary(self, labels: np.array, metric: T) -> float:
        """Return value of chosen metric.

        Parameters
        -----------
        labels: np.array
            Labels  (cluster number for every observation in dataset).
        metric: T
            loaded metric.

        Returns
        -----------
        float
            Value of metric.
        """
        return metric(self.data, labels)
