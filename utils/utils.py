import pickle
import pyclustering
import numpy as np
import logging
import re
import yaml
from pathlib import Path
from typing import List, Dict, Union, Any, NoReturn, Tuple
from sluch_ai.src.cfg import CFG
from watcher.src.logger import LOGGING_CONFIG
import io
import os
from scipy.io import wavfile
from pydub import AudioSegment
import json
import librosa

def read_config(path: str) -> Dict:
    """Load config based on provided filepath.

    Param
    ---------
    path: str
        Path to provided file.

    Returns
    ---------
    Dict
        configuration file in a dictionary form.
    """
    with open(path, "rb") as confile:
        config = yaml.safe_load(confile)
    return config

def join_output(input: List[List]) -> str:
    """Return string based on chunk data which is an List of lists.

    Params
    --------
    input: List[List]
        self.model output which is an list
        of list in case of chunked data.

    Returns
    --------

    results: str
        Joined data.
    """
    results: str = "/".join([el[0].lower() for el in input])
    return results


def loop_records(dt, tr, sr, logger):
    n_files = len(dt)
    logger.info(f"Processing for {n_files} files.")
    for ix in range(n_files):
        try:
            logger.info(f"Start processing from {ix} of {n_files}")
            sliced = sr.slice_wav(file=dt[ix][0], chunk_size=8, sr=16_000)
            output = sr.batch_inference(files=sliced, sr=16_000)
            del sliced
            _res = {dt.files[ix]: output}
            tr.add(_res)
            if ix % 10 == 0:
                tr.save_storage()
        except Exception as e:
            print(e)
            logger.info(f"{e}")
    tr.save_storage()


def single_record(dt, tr, sr, file: str):
    fileslist = dt.gather_files()
    if file not in fileslist:
        clean_out = dt.load_storage()[file]
    else:
        try:       
            wav, sr = librosa.load(file, sr=CFG.sampling_rate)
            sliced = sr.slice_wav(file=wav, chunk_size=8, sr=CFG.sampling_rate)
            output = sr.batch_inference(files=sliced, sr=CFG.sampling_rate)
            del sliced
            _res = {file: output}
            tr.add(_res)
            tr.save_storage()
        except:
            print("Transcription failed")
            clean_out = ""
            
    return clean_out

def logger_initialize(name: str):
    """Initialize logger object"""
    logging.root.setLevel(logging.NOTSET)
    logging.basicConfig(level=logging.NOTSET)
    logging.config.dictConfig(LOGGING_CONFIG)
    log = logging.getLogger(name)
    return log


def find_keywords(input_: str, rules: Dict, cat_name: str) -> List:
    """Find all keywords defined in rules dict

    Parameters
    ----------
    input_: str
        String in which we will find keywords
    rules: Dict
        Dictionary which contains all keywords grouped by category
    cat_name: str
        One from list of categories in rules dict

    Returns
    -------
    List with all keywords found in string
    """
    _class = rules.get(cat_name)
    regex = re.compile("(%s)" % "|".join(map(re.escape, _class)))

    return re.findall(regex, input_)


def keywords_loop(data: Dict, rules: Dict) -> Dict:
    """Create dict with all keywords find in a set of audio files

    Parameters
    ----------
    data: Dict
        Dictionary with paths and transcriptions for a set of audio files
    rules: Dict
        Dictionary which contains all keywords grouped by category

    Returns
    -------
    Dictionary with all found keywords in transcription
    """
    results = dict()
    for (path, records) in data.items():
        result = dict()
        for (k, v) in rules.items():
            a = list(set(find_keywords(input_=records, rules=rules, cat_name=k)))
            if len(a):
                result[k] = a
        results[path] = result
    return results

def convert_wav_to_byte(wav: np.array, sampling_rate: int = 16000):
    bytes_wav = bytes()
    byte_io = io.BytesIO(bytes_wav)
    wavfile.write(byte_io, sampling_rate, wav)
    wav_cut_bytes = byte_io.read()
    return wav_cut_bytes

def convert_wav_to_audiosegment(wav, sr=16000):
    return AudioSegment(wav, frame_rate=sr, sample_width=wav.dtype.itemsize, channels=1)


def initialize_logger(name: str = "logger_name"):
    """Initialize logger object
    based on predefined watcher.src.logger arch.
    """
    logging.root.setLevel(logging.NOTSET)
    logging.basicConfig(level=logging.NOTSET)
    logging.config.dictConfig(LOGGING_CONFIG)
    log = logging.getLogger(name)
    return log


def get_clusters_info(data: Tuple[List, np.array], clusters:List) -> Dict:
    """ Return clusters with information
        for which cluster every information
        belongs to.

    Params
    ---------
    data: Tuple[List, np.array]
        Input data for which algorithms is train.

    clusters: pyclustering.cluster
        pyclustering object, trained algorithm.

    Returns
    ---------

    results: Dict
        Output dictionary in which key is number 
        of cluster  and value is an name of files.
    """
    results: Dict = {}
    n_clusters = len(clusters)
    if isinstance(data, List):
        data = np.array(data)
    for el in range(n_clusters):
        results[el] = data[clusters[el]] 
    return results

def save_pkl_object(object: Any, filename: Union[str, Path]) -> None:
    with open(filename, 'wb') as pklfile:
        pickle.dump(object, pklfile)
