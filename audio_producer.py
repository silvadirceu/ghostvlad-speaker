from __future__ import division
from __future__ import print_function
import json
import multiprocessing as mp

import os
import random
import numpy as np
import psutil
import deepdish as dd
from joblib import Parallel, delayed

from load_audio import load_data
from utils import log

import time

_LOG_FILE_PATH = "extractor.log"
_LOG_FILE = log(_LOG_FILE_PATH)
_ERRORS = list()


def feature_path(audio_path, feature_dir):
    path = audio_path.split('/')
    filename = path[-1].split(".")[0]
    session = path[-2]
    speaker_id = path[-3]

    work_dir = os.path.join(feature_dir,speaker_id,session)

    filepath = os.path.join(work_dir, filename + '.h5')

    return work_dir, filepath


def split_list_with_N_elements(seq,n):
    #spli a list in sublists with n elements + the reminder

    newlist = [seq[i * n:(i + 1) * n] for i in range((len(seq) + n - 1) // n )]
    return newlist

class FeatureExtractor(object):

    def __init__(self, data_json, feature_dir=None, batch_size=32, sample_rate=16000,
                 min_duration=600, max_duration=2500, mode="train",
                 n_workers=-1, parallel=True,
                 save_features=True):
        """
        Args:
            data_json : json format file with speech data.
                        'path', the path of the wave file.
                        'spkid', the speaker's identity in int.
            batch_size : Size of the batches for training.
            sample_rate : Rate to resample audio prior to feature computation.
            min_duration : Minimum length of audio sample in milliseconds.
            max_duration : Maximum length of audio sample in milliseconds.
        """

        self.params ={"sample_rate": sample_rate,
                      "min_duration": min_duration,
                      "max_duration": max_duration,
                      "feature_dir": feature_dir,
                      "mode": mode,
                      "save_feature": save_features,
                      "parallel": parallel,
                      "batch_size": batch_size}

        with open(data_json, 'r') as fid:
            self.data = json.load(fid)

        if n_workers == -1:
            self.params["num_cpus"] = psutil.cpu_count(logical=False)
        else:
            self.params["num_cpus"] = n_workers


    def compute_features(self):

        if self.params["parallel"]:

            #batches = [self.data[i:i + self.params["batch_size"]]
            #           for i in range(0, len(self.data) - self.params["batch_size"] + 1, self.params["batch_size"])]

            batches = split_list_with_N_elements(self.data, self.params["batch_size"])

            #batches = np.array_split(self.data, self.params["batch_size"])
            start = time.time()

            Parallel(n_jobs=self.params["num_cpus"], verbose=100)(
                delayed(self._extractor)(cpath, self.params) for cpath in batches)

            stop=time.time()
            print("time: ", stop-start)

        else:
            start = time.time()
            self._extractor(self.data[:320], self.params)
            stop=time.time()
            print("time: ", stop-start)

    def _extractor(self, list_path, params):

        rand_duration = np.random.randint(params["min_duration"], params["max_duration"])
        utterance_spec = None

        for file in list_path:

            audio_path = file["path"]

            try:
                utterance_spec = load_data(audio_path,
                                           sr=params["sample_rate"],
                                           mode=params["mode"])
            except:
                _ERRORS.append(audio_path)
                _LOG_FILE.debug("Error: skipping computing features for audio file --%s-- " % audio_path)
                utterance_spec = None

            work_dir, filepath = feature_path(audio_path, params["feature_dir"])

            if utterance_spec is not None:
                try:
                    if not os.path.exists(work_dir):
                        os.makedirs(work_dir)

                    dd.io.save(filepath, utterance_spec)
                except:
                    _ERRORS.append(filepath)
                    _LOG_FILE.debug("Error: saving features for audio file --%s-- " % filepath)


class AudioProducer(object):

    def __init__(self, data_json, batch_size, sample_rate=16000,
                 min_duration=600, max_duration=2500):
        """
        Args:
            data_json : json format file with speech data.
                        'path', the path of the wave file.
                        'spkid', the speaker's identity in int.
            batch_size : Size of the batches for training.
            sample_rate : Rate to resample audio prior to feature computation.
            min_duration : Minimum length of audio sample in milliseconds.
            max_duration : Maximum length of audio sample in milliseconds.
        """
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        with open(data_json, 'r') as fid:
            self.data = json.load(fid)


    def queue_featurize(self, consumer, producer, sample_rate):
        while True:
            try:
                batch = consumer.get(block=True, timeout=5)
            except mp.queues.Empty as e:
                print("queue_featurize finished or error happened.")
                return
            labels = []
            inputs = []
            rand_duration = np.random.randint(self.min_duration, self.max_duration)
            minSpec = rand_duration
            utterances = []
            for b in batch:
                labels.append(int(b['spkid']))
                utterance_spec = load_data(b['path'], 
                                            sr=sample_rate,
                                            rand_duration=rand_duration,
                                            is_training=True)
                if(utterance_spec is None): # Error loading some audio file.
                    break
                if(utterance_spec.shape[1]<minSpec): # align to mini audio file.
                    minSpec = utterance_spec.shape[1]
                utterances.append(utterance_spec)

            if(len(utterances)!=len(batch)): # Error loading some audio file.
                continue
            else:
                for utterance_spec in utterances:
                    inputs.append(np.expand_dims(utterance_spec[:,:minSpec], -1))

            producer.put((np.array(inputs), labels))


    def iterator(self, max_size=500, num_workers=10):
        random.shuffle(self.data)
        batches = [self.data[i:i+self.batch_size]
                   for i in range(0, len(self.data) - self.batch_size + 1, self.batch_size)]

        consumer = mp.Queue()
        producer = mp.Queue(max_size)
        for b in batches:
            consumer.put(b)

        procs = [mp.Process(target=self.queue_featurize,
                            args=(consumer, producer,
                                  self.sample_rate))
                 for _ in range(num_workers)]
        for p in procs:
            p.start()

        for _ in batches:
            yield producer.get()


