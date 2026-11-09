import argparse # read terminal options
import csv # read the dataset
import json # read the datset
import os # get system info
import time # used to time the run
from multiprocessing import Pool # allows multithreading
from pathlib import Path # a simplier way of handling paths

import librosa # used for the spectograms
import numpy as np # optimize spectogram math
import soundfile as sf # reads audio files

import tokenizer # imports the tokenizer.py file