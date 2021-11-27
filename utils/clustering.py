import numpy as np 
import os 
import yaml 
import pickle
import pyclustering
import pyclustering.cluster
import sklearn.cluster 

from pathlib import Path
from pyclustering.cluster import cluster_visualizer
from typing import List, Dict
from functools import wraps
from time import time
from typing import Tuple, Any
from ..src.cfg import ConfigPyclustering

def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        ts = time()
        result = f(*args, **kwargs)
        te = time()
        print(f'func: {f.__name__} takes: {te-ts:.2f}s')
        return result
    return wrap

def read_config(path: str = 'config/config.yaml') -> Dict:
    """ Load configuration file from path.

    Parameters
    -----------
    path : str
    Loaded file full path

    Returns 
    -----------
    Dict 
        Dictionary with proper parameters.

    """
    with open(path, 'r') as confile:
        config = yaml.safe_load(confile)
    return config

def list_files(path: str, extension: str) -> List:
    _files = list(Path(path).rglob(f'*{extension}'))
    assert len(_files) > 0, 'Number of files should be greater than 0'
    return _files 

def concat_iters(_iter: np.ndarray):
    _tmp = _iter.reshape(-1)
    return _tmp 

def pca_convert(_data: np.array) -> np.array:
    """ One case scenario, I dont think that it should be reusable
        Move from test_pca to avoid repetition 
        Load np.array, concatenate it to proper shape
        Flatten data

    Parameters 
    ----------  
    data: np.array 

    Returns
    ----------
    np.array 
        Reshaped data
    """
    n_obs = len(_data.values)
    assert n_obs > 0, 'Number of observation should be greater than zero'
    _values = list(_data.values)
    _concat = np.concatenate(_values).ravel().reshape(n_obs, -1)
    return _concat
    
def training_decor(func):
    """Decor for training procedure, stricly fitted to unsupervised methods

    Parameters
    ----------
        func (method): 

    Returns
    ----------
        wrapper object
    """
    def wrapper(self, *args, **kwargs):
        logger = kwargs.get('logger')
        logger.debug("Clusterization based on pca com")
        logger.info("Start training process")
        try:
            func(self, *args, **kwargs) 
        except Exception as _error:
            logger.critical(_error, exc_info = True )
    return wrapper

def exception_handler(func):
    """ Decorator function which extend logger information
        if any exception appears
    Parameters
    ----------
    func : class.method

    Returns 
    --------
    wrapper : object
    """
    def wrapper(self,  *args, **kwargs):
        logger = kwargs.get('logger')
        try:
            func(self, *args, **kwargs)
        except Exception as _error:
            logger.critical(_error, exc_info = True)
    return wrapper

def load_data(path: str) -> np.array:
    assert os.path.exists(path), 'Provided path does not exist'
    with open(path, 'rb') as file:
        _data = pickle.load(file)
    return _data

def initialize_colors(n_cols : int) -> List:
    """Create list with random colors which lenght is equal to n_cols value.

    Params
    -------
    n_cols : int
        Number of colors which will be produces.

    Returns
    --------
    List
        List of length n_cols which is filled with tuples of shape (3,)
    """
    _colors = tuple(np.random.choice(range(256), size = 3) for el in range(n_cols))
    return _colors

def sort_values_dict(_dict : Dict) -> Dict:
    return sorted(_dict.items(), key = lambda item: item[1], reverse = True)

def perc_values(_dict : Dict) -> Dict:
    return {k : v/len(_dict.values()) for k, v in _dict.items()}


# Pyclust trainer helper functions

def listable_pca(data: np.ndarray):
    """ Return list instead of np.ndarray structure.a
    Parameters
    -----------
    data :np.ndarray:
        Data in np.ndarray representation.

    Return
    -------
    List:
        Data in List form.
    """
    return [list(el) for el in data]

def create_subdict(data: List):
    """ Return data in List form as dictionary, 
        Start, to which model parameters is appended. 
    Parameters
    -----------
    data: List
        Data (potentially pca component output in 2d form).

    Return 
    -------
        Dict:
            Dictionary with 'data' key. 
    """
    return {'data' : data}

def subset_dict(method: str, data: List, config: ConfigPyclustering):
    """ add parameters to predefined dictionary. 
        Usable in initialize pyclustering methods. 

    Parameters
    -----------
    method: str
        Name of unsupervised method, which should exist in pyclustering configuration file.
    """
    assert getattr(config, method) is not None, 'Provided keys does not exist'
    subdict = create_subdict(data = data)
    tmp_dict: Dict = {'method' : getattr(config, method)}
    subdict.update(tmp_dict)
    return subdict

def get_clusters_vis(clusters: List, data: Tuple[List, np.array]):
    """
    visualizer = cluster_visualizer()
    clusters = instance.get_clusters()
    visualizer.append_cluster(clusters, data)
    """
    if isinstance(data, List):
        data = np.array(data)
    visualizer = cluster_visualizer()
    visualizer.append_clusters(clusters, data)
    visualizer.show()

def aggregate_data_pca(sp: Any) -> Dict:
    """One case scenario. Used to properly initialized pca 
       data transformation with sluch_ai.clustering.pca_prep
       Aggregate wav file to dictionary, key state as a filename.

       Params 
       --------
       sp: SoundPreprocess instance

       Return
       -------
       Dict {str: np.ndarray}
    """
    n_files = len(sp)
    _results: Dict = {}
    for el in range(n_files):
        _tmp: Tuple = sp[el]
        _tmp_wav: np.ndarray = _tmp[0] 
        _tmp_filename: str = _tmp[1]
        _results[_tmp_filename] = _tmp_wav
    return _results