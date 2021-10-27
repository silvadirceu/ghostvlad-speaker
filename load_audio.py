# Third Party
import librosa
import numpy as np
from pydub import AudioSegment
import io

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


