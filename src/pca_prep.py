
import numpy as np
import os
import pickle
import re
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Union, TypeVar
from ...src.cfg import ConfigPCA

def save_pkl(obj: np.ndarray, filepath: str) -> None:
    """
    Save .pkl file based on previously defined .wav files, 
    presented as np.ndarray object. 

    Params
    -------
    obj: np.ndarray 
        Output from slucha_ai.utils.clustering.aggregate_data_pca function. 

    filepath: str
        Filepath in which PCA.object will be saved. 

    Returns
    -------
    None : 
    """
    with open(filepath, "wb") as pklfile:
        pickle.dump(obj, pklfile)

def save_data(obj: Dict, filepath: str) -> None:
    with open(filepath, "wb") as file:
        pickle.dump(obj, filepath)

def init_pca(data: np.ndarray, n_components: int, scaling: bool, save: bool) -> None:
    """Initialize principal component analysis

    Args:
        data (np.ndarray): unscaled data
        n_components (20): number of principal component analysis features
        config (Dict): path to configuration file
        scaling (bool, optional): using StandardScaler or not. Defaults to True.
        save (bool, optional): saving scaled data. Defaults to True.
    """
    cfg = ConfigPCA() 
    if scaling:
        sc = StandardScaler()
        data = sc.fit_transform(data)
    _pca = PCA(n_components=n_components)
    _data_pca = _pca.fit_transform(data)
    if save:
        save_pkl(obj=_data_pca, filepath= cfg.filepath)
    save_pkl(_pca, cfg.pca_object_path)
