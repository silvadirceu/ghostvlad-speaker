#https://www.pyimagesearch.com/2021/06/28/data-augmentation-with-tf-data-and-tensorflow/

# import the necessary packages

import tensorflow as tf
import tensorflow_io as tfio
import os

import librosa
import numpy as np
from pydub import AudioSegment
import io


def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)

  # Note: You'll use indexing here instead of tuple unpacking to enable this
  # to work in a TensorFlow graph.
  return parts[-2]

def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

def load_audio(imagePath):
    # read the image from disk, decode it, convert the data type to
    # floating point, and resize it
    image = tf.io.read_file(imagePath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, (156, 156))
    # parse the class label from the file path
    label = tf.strings.split(imagePath, os.path.sep)[-2]

    # return the image and the label
    return (image, label)

def augment_using_layers(images, labels, aug):
	# pass a batch of images through our data augmentation pipeline
	# and return the augmented images
	images = aug(images)
	# return the image and the label
	return (images, labels)


def augment_using_ops(images, labels):
	# randomly flip the images horizontally, randomly flip the images
	# vertically, and rotate the images by 90 degrees in the counter
	# clockwise direction
	images = tf.image.random_flip_left_right(images)
	images = tf.image.random_flip_up_down(images)
	images = tf.image.rot90(images)
	# return the image and the label
	return (images, labels)


# load from m4a, the format of voxcele2 dataset
def load_m4a(vid_path, sr):
    audio = AudioSegment.from_file(vid_path, "mp4")
    audio = audio.set_frame_rate(sr)
    audio = audio.set_channels(1)
    audio = audio.set_sample_width(2)
    buf = io.BytesIO()
    audio.export(buf, format='s16le')
    wav = np.frombuffer(buf.getbuffer(), np.int16)
    wav = np.array(wav/32768.0, dtype=np.float32)

    intervals = librosa.effects.split(wav, top_db=20)
    wav_output = []
    for sliced in intervals:
      wav_output.extend(wav[sliced[0]:sliced[1]])
    wav_output = np.array(wav_output)
    return wav_output

# ===============================================
#       code from Arsha for loading data.
# ===============================================
"""
def load_wav(vid_path, sr, mode='train'):
    wav, sr_ret = librosa.load(vid_path, sr=sr)
    assert sr_ret == sr
    if mode == 'train':
        extended_wav = np.append(wav, wav)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
        return extended_wav
    else:
        extended_wav = np.append(wav, wav[::-1])
        return extended_wav
"""

def load_wav(vid_path, sr):
    wav, sr_ret = librosa.load(vid_path, sr=sr)
    assert sr_ret == sr

    intervals = librosa.effects.split(wav, top_db=20)
    wav_output = []
    for sliced in intervals:
      wav_output.extend(wav[sliced[0]:sliced[1]])
    wav_output = np.array(wav_output)
    return wav_output


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T


def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=250, mode='train'):

    try:
        if(path.endswith('.wav')):
            wav = load_wav(path, sr=sr)
        elif(path.endswith('.m4a')):
            wav = load_m4a(path, sr=sr)
        else:
            print("!!! Not supported audio format.")
            return None
    except Exception as e:
        print("Exception happened when load_data('{}'): {}".format(path, str(e)))
        return None

    #wav = load_wav(path, sr=sr, mode=mode)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    if mode == 'train':
        randtime = np.random.randint(0, time-spec_len)
        spec_mag = mag_T[:, randtime:randtime+spec_len]
    else:
        spec_mag = mag_T
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    return (spec_mag - mu) / (std + 1e-5)