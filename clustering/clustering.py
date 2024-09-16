"""
imports
"""

import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import DBSCAN

"""
audio settings
"""

audio_file_path = "D:/Data/audio/Gutenberg/Night_and_Day_by_Virginia_Woolf_48khz.wav"
audio_sample_rate = 48000
#audio_excerpt_length = 1000 # in milisecs
#audio_excerpt_offset = 500 # in milisecs
audio_excerpt_length = 30 # in milisecs
audio_excerpt_offset = 15 # in milisecs

"""
load audio
"""

audio_waveform, audio_sample_rate = librosa.load(audio_file_path, sr=audio_sample_rate)
audio_waveform_sc = audio_waveform.shape[0]

audio_waveform_sc

"""
create audio excerpts
"""

audio_excerpts = []

audio_excerpt_length_sc = int(audio_excerpt_length / 1000 * audio_sample_rate)
audio_excerpt_offset_sc = int(audio_excerpt_offset / 1000 * audio_sample_rate)

for sI in range(0, audio_waveform_sc - audio_excerpt_length_sc, audio_excerpt_offset_sc):
    
    audio_excerpt = audio_waveform[sI:sI + audio_excerpt_length_sc]
    audio_excerpts.append(audio_excerpt)
    
audio_excerpts = np.stack(audio_excerpts, axis=0)


"""
calculate audio features
"""

audio_features = {}

# audio waveforms

audio_features["waveform"] = audio_excerpts

# rms

root_mean_square = []

for aI, audio_excerpt in enumerate(audio_excerpts):
    
    print("aI ", aI, " out of ", len(audio_excerpts))

    rms = librosa.feature.rms(y=audio_excerpt)
    rms = rms.flatten()
    
    root_mean_square.append(rms)

root_mean_square = np.stack(root_mean_square, axis=0)

audio_features["root mean square"] = root_mean_square

# spectral centroid

spectral_centroid = []

for audio_excerpt in audio_excerpts:

    cent = librosa.feature.spectral_centroid(y=audio_excerpt, sr=audio_sample_rate)
    cent = cent.flatten()
    
    spectral_centroid.append(cent)

spectral_centroid = np.stack(spectral_centroid, axis=0)

audio_features["spectral centroid"] = spectral_centroid

# spectral bandwidth

spectral_bandwidth = []

for audio_excerpt in audio_excerpts:

    bandwidth = librosa.feature.spectral_bandwidth(y=audio_excerpt, sr=audio_sample_rate)
    bandwidth = bandwidth.flatten()
    
    spectral_bandwidth.append(bandwidth)

spectral_bandwidth = np.stack(spectral_bandwidth, axis=0)

audio_features["spectral bandwidth"] = spectral_bandwidth

# spectral contrast

spectral_contrast = []

for audio_excerpt in audio_excerpts:

    contrast = librosa.feature.spectral_contrast(y=audio_excerpt, sr=audio_sample_rate)
    contrast = contrast.flatten()
    
    spectral_contrast.append(contrast)

spectral_contrast = np.stack(spectral_contrast, axis=0)

audio_features["spectral contrast"] = spectral_contrast

"""
KMeans Clustering
"""


from sklearn.cluster import KMeans

cluster_count = 10
random_state = 170

km = KMeans(n_clusters=cluster_count, n_init= "auto", random_state = random_state)
#labels =  km.fit_predict(audio_features["spectral centroid"])
#labels =  km.fit_predict(audio_features["spectral bandwidth"])
#labels =  km.fit_predict(audio_features["spectral contrast"])
labels =  km.fit_predict(audio_features["root mean square"])

"""
concatenate audio files per cluster
"""

# calculate number of audio excerpts per cluster
audio_count_per_cluster, _ = np.histogram(labels, bins=cluster_count)

# prepare empty audio_cluster waveforms
audio_clusters = []

for cI in range(cluster_count):
    
    audio_count = audio_count_per_cluster[cI]

    if audio_count > 0:
        audio_cluster_sc = audio_excerpt_length_sc + audio_excerpt_offset_sc * (audio_count - 1)
    else:
        audio_cluster_sc = 1
        
    audio_clusters.append(np.zeros(audio_cluster_sc, dtype=np.float32))
    
# add audio excerpts to audio_cluster waveforms

audio_window_envelope = np.hanning(audio_excerpt_length_sc)

audio_cluster_insert_indices = np.zeros(cluster_count, dtype=np.int32)

for excerpt_index, label in enumerate(labels):
    
    audio_excerpt = audio_excerpts[excerpt_index]
    audio_cluster_insert_index = audio_cluster_insert_indices[label]
    audio_cluster_waveform = audio_clusters[label]
    
    audio_cluster_waveform[audio_cluster_insert_index:audio_cluster_insert_index+audio_excerpt_length_sc] += audio_excerpt * audio_window_envelope
    
    audio_cluster_insert_indices[label] += audio_excerpt_offset_sc

# save audio_cluster waveforms

for cI in range(cluster_count):
    
    audio_file_name = "root_mean_square_30ms_cluster_{}.wav".format(cI)
    audio_cluster_waveform = audio_clusters[cI]
    
    sf.write(audio_file_name, audio_cluster_waveform, audio_sample_rate)

