import pickle as pkl
from collections import deque
from pyclustering.cluster import birch
from sluch_ai.clustering.clustering.clustering import PyClustTrainer
from sluch_ai.clustering.src.preprocess import SoundPreprocess
from sluch_ai.utils.clustering import list_files
from sluch_ai.clustering.src.pca_prep import init_pca
from sluch_ai.src.cfg import ConfigPyclustering
from sluch_ai.src.cfg import ConfigPCA

def _iter(method: str):
    return  PyClustTrainer(methodname = method)._train()

def main():

    cfgp = dir(ConfigPyclustering)
    _attributes = [el for el in cfgp if not (el.startswith('__') or el == 'pyclustering')]   
    _objs = map(_iter, _attributes)
    #https://docs.python.org/3/library/collections.html
    deque(_objs)

if __name__ == "__main__":
    main()
