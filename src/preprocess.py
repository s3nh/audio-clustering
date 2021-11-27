import gc
import numpy as np
import os
import librosa
import torch

from librosa.feature import melspectrogram, mfcc
from ...utils.clustering import read_config, list_files, concat_iters
from ...utils.utils import initialize_logger
from ...src.cfg import ConfigSound
from typing import Tuple 

class SoundPreprocess:
    def __init__(self):
        self.logger = initialize_logger(name = 'soundpreprocess')
        self.config = ConfigSound()
        self.path = self.config.data_path
        self.files = self.get_files()
        self.n_files = len(self.files)
        self.nth = 0

    def load_file(self, path: str) -> Tuple[np.ndarray, int]:
        """Load file based on its path

        Args:
            path (str): Path to file

        Returns:
            data (np.array): wav form of loaded file
            sr (int) : sample rate
        """
        data, sr = librosa.load(path)
        return data, sr

    def get_files(self):
        """Get files to process

        Returns:
            List: List with full path files with provideed extension
        """
        _files = list_files(self.path, extension=self.config.extension)
        return _files

    def get_vocabconfig(self):
        """List vocab_config keys
            Helper like to avoid spotting into .yaml files

        Returns:
            [Dict]: [Keys for vocab config]
        """
        return self.vocab_config.keys()

    def get_config(self):
        """List config.keys
            Helper like function to avoid spotting into .yaml files

        Returns:
            [Dict]: [Keys for configuration file]
        """
        return self.config.keys()

    def create_melspectrogram(self, file: np.ndarray):
        """Create melspectrgram from listed file
            based on params provided in self.vocab_config:
        Returns:
            [np.array]: Melspectrogram with maximum frequency and number of mels
            based on vocab_config dictionary values
        """
        return melspectrogram(
            file, n_mels=self.config.n_mels, fmax= self.config.f_max
        )

    def create_mfcc(self, file: np.ndarray, sr: int = 16000):
        """ Create mfcc from source data """
        return mfcc(file, sr=sr)

    def pad(self, file: np.array, constant_length: int) -> np.array:
        if constant_length > len(file):
            return np.pad(file, (0, constant_length - len(file)))
        else:
            return file[:constant_length]

    def __cache__(self):
        """Clear cache if needed"""
        gc.collect()

    def __getitem__(self, ix: int):
        """Get item and return its processed form
        Args:
            ix: int - index of processed file
        Returns:
            processed file

        Concat iters should be provided before padding to override create an enormous arrays
        """
        _data, _ = self.load_file(self.files[ix])
        _spectro = self.create_melspectrogram(_data)
        _spectro = concat_iters(_spectro)
        _spectro = self.pad(_spectro, constant_length=self.config.max_len)
        return _spectro, self.files[ix]

    def __len__(self):
        """Return length of processed files
        Returns:
            [int]: number of files to process
        """
        return len(self.files)
