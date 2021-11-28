from functools import wraps
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path, PurePath
import pickle
import pyclustering.cluster
import sklearn
import sklearn.cluster

from typing import Dict, Union, Optional
from ...utils.clustering import listable_pca
from ...utils.clustering import subset_dict
from ...utils.clustering import timing
from ...utils.utils import initialize_logger
from ...src.cfg import ConfigPCA, ConfigPyclustering

def init_method(config: Dict, key: str):
    """[summary]

    Args:
        config (Dict): [description]
        key (str): [description]

    Returns:
        [type]: [description]
    """
    attr = getattr(sklearn.cluster, key)
    return attr(**config.get(key))

class PcaDataLoader(object):
    """Load Principal Component Analysis decomposed Data.
    Parameters
    -----------
    path : str
        Transformed data filepath
    """

    def __init__(self):
        self.config = ConfigPCA()
        self.path = self.config.filepath 
        self.data = self.load_data()

    def load_data(self):
        with open(self.path, "rb") as confile:
            _data = pickle.load(confile)
        return _data


class PyClustTrainer(PcaDataLoader):
    def __init__(self, methodname: str):
        self.import_modules(method = methodname)
        self.logger = initialize_logger(name="PyclustTrainer")
        self.config = ConfigPyclustering() 
        self.methodname = methodname
        self.mod = pyclustering.cluster.__getattribute__(self.methodname)
        self.method = getattr(self.mod, self.methodname)
        self.pyclust: bool = True 
        self.data = PcaDataLoader().data

    def init_pyclust(self):
        return True if self.config.pyclustering is not None else False

    def import_modules(self, method: str):
        __import__(f"pyclustering.cluster.{method}")

    def get_methods(self) -> Dict:
        """Return unsupervised methods names.
        Returns
        -------
        Dict
            dict keys
        """
        return self.config.keys()

    def get_params(self, methodname: str) -> Dict:
        """Return parameters for proper method.
        Parameters
        ----------
        methods: str
            Name of method which explicitly respond to self.config.keys
        Returns
        ----------
        Dict
            dictionary which contain listed method parameters
        assert methodname in self.config.keys(),  f'{methodname} is not included in config file'
        return self.config.get(methodname)
        """
        raise NotImplementedError

    def init_method(self, methodname: str):
        """Initialize method based on its name.
        Parameters
        ----------
        methodname: str
            Name of processed method.

        Returns
        ----------
        object
            Class instance, either sklearn.cluster or pysclustering, dependent
            on self.pyclust value.
        """
        if not self.pyclust:
            _attr = self.getattr_(methodname)
            return _attr(**self.config.methodname)
        else:
            __import__(pyclustering.cluster, methodname)
            _attr = self.method
            return _attr

    def save_model(self, instance):
        """Save object to .pkl file. Include parameters in its naming.
        Parameters
        ----------
            *args
            **kwargs
        Returns
        ----------
        """
        Path("saved_models").mkdir(exist_ok = True)
        _path = Path("saved_models").joinpath(self.methodname + "_" + ".pkl")
        with open(_path, "wb") as modfile:
            pickle.dump(instance.clusters, modfile)
        self.logger.info(f"File saved in {_path}")

    def fit_(self) -> None:
        """Fit data dependent on used method.
        Parameters
        ----------
        method : str
            Method name (self.config file key).
        Returns
        ----------
        None
            Fit algorithm on self.data object.
        """
        if self.pyclust:
            self.data = listable_pca(self.data)
            _params = subset_dict(method=self.methodname, data=self.data, config=self.config)
            _instance = self.method(data = _params['data'], **_params['method'])
            _instance.process()
        else:
            _instance = self.method(*_params.values())
            _instance.fit(self.data)
        return _instance

    def getattr_(self, methodname: str):
        """Return feature attributes based on config file.
        Parameters
        ----------
        methodname : str
            Name of proceed method.

        Returns
        ----------
        type
            Called method instance.
        """
        if self.pyclust:
            self.import_modules(method=methodname)
            return getattr(pyclustering.cluster, methodname)
        else:
            return getattr(sklearn.cluster, methodname)

    def _train(self, **kwargs):
        """Fit algorithm to self.data and save it to file.
        Keyword Parameters
        -------------------
        methodname: str
        Name of processed algorithm

        Returns
        -----------
        None
        """
        self.logger.info(
            f"Initialize method {self.methodname}"
        )
        self.logger.info(f"Start fitting data \n")
        _instance = self.fit_()
        self.save_model(instance = _instance)
        self.logger.info(f"Passed data fit")
