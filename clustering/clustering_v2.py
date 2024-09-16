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
audio_excerpt_length = 1000 # in milisecs
audio_excerpt_offset = 900 # in milisecs

#audio_excerpt_length = 40 # in milisecs
#audio_excerpt_offset = 30 # in milisecs

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

"""
root_mean_square = []

for audio_excerpt in audio_excerpts:
    
    rms = librosa.feature.rms(y=audio_excerpt)
    rms = rms.flatten()
    
    root_mean_square.append(rms)

root_mean_square = np.stack(root_mean_square, axis=0)

audio_features["root mean square"] = root_mean_square
"""

"""
# chroma stft

chroma_stft = []

for audio_excerpt in audio_excerpts:

    stft = librosa.feature.chroma_stft(y=audio_excerpt, sr=audio_sample_rate)
    stft = stft.flatten()
    
    chroma_stft.append(stft)

chroma_stft = np.stack(chroma_stft, axis=0)

audio_features["chroma stft"] = chroma_stft
"""

"""
# chroma cqt (super slow to calculate)

chroma_cqt = []

for aI, audio_excerpt in enumerate(audio_excerpts):
    
    print("aI ", aI, " out of ", audio_excerpts.shape[0])

    cqt = librosa.feature.chroma_cqt(y=audio_excerpt, sr=audio_sample_rate)
    cqt = cqt.flatten()
    
    chroma_cqt.append(cqt)

chroma_cqt = np.stack(chroma_cqt, axis=0)

audio_features["chroma cqt"] = chroma_cqt
"""

"""
# chroma cens (super slow to calculate)

chroma_cens = []

for aI, audio_excerpt in enumerate(audio_excerpts):
    
    print("aI ", aI, " out of ", audio_excerpts.shape[0])

    cens = librosa.feature.chroma_cens(y=audio_excerpt, sr=audio_sample_rate)
    cens = cens.flatten()
    
    chroma_cens.append(cens)

chroma_cens = np.stack(chroma_cens, axis=0)

audio_features["chroma cens"] = chroma_cens
"""

"""
# chroma vqt (slow to calculate)

chroma_vqt = []

for aI, audio_excerpt in enumerate(audio_excerpts):
    
    print("aI ", aI, " out of ", audio_excerpts.shape[0])

    vqt = librosa.feature.chroma_vqt(y=audio_excerpt, sr=audio_sample_rate, intervals="ji5")
    vqt = vqt.flatten()
    
    chroma_vqt.append(vqt)

chroma_vqt = np.stack(chroma_vqt, axis=0)

audio_features["chroma vqt"] = chroma_vqt
"""

"""
# mel spectrogram

mel_spectrogram = []

for aI, audio_excerpt in enumerate(audio_excerpts):
    
    print("aI ", aI, " out of ", audio_excerpts.shape[0])

    mel = librosa.feature.melspectrogram(y=audio_excerpt, sr=audio_sample_rate)
    mel = mel.flatten()
    
    mel_spectrogram.append(mel)

mel_spectrogram = np.stack(mel_spectrogram, axis=0)

audio_features["mel spectrogram"] = mel_spectrogram
"""

"""
# mfcc (cool)

mfcc = []

for aI, audio_excerpt in enumerate(audio_excerpts):
    
    print("aI ", aI, " out of ", audio_excerpts.shape[0])

    mfcc_ = librosa.feature.mfcc(y=audio_excerpt, sr=audio_sample_rate)
    mfcc_ = mfcc_.flatten()
    
    mfcc.append(mfcc_)

mfcc = np.stack(mfcc, axis=0)

audio_features["mfcc"] = mfcc
"""

"""
# spectral centroid

spectral_centroid = []

for audio_excerpt in audio_excerpts:

    cent = librosa.feature.spectral_centroid(y=audio_excerpt, sr=audio_sample_rate)
    cent = cent.flatten()
    
    spectral_centroid.append(cent)

spectral_centroid = np.stack(spectral_centroid, axis=0)

audio_features["spectral centroid"] = spectral_centroid
"""

"""
# spectral bandwidth

spectral_bandwidth = []

for audio_excerpt in audio_excerpts:

    bandwidth = librosa.feature.spectral_bandwidth(y=audio_excerpt, sr=audio_sample_rate)
    bandwidth = bandwidth.flatten()
    
    spectral_bandwidth.append(bandwidth)

spectral_bandwidth = np.stack(spectral_bandwidth, axis=0)

audio_features["spectral bandwidth"] = spectral_bandwidth
"""

"""
# spectral contrast
# High contrast values generally correspond to clear, narrow-band signals, while low contrast values correspond to broad-band noise. 

spectral_contrast = []

for audio_excerpt in audio_excerpts:

    contrast = librosa.feature.spectral_contrast(y=audio_excerpt, sr=audio_sample_rate)
    contrast = contrast.flatten()
    
    spectral_contrast.append(contrast)

spectral_contrast = np.stack(spectral_contrast, axis=0)

audio_features["spectral contrast"] = spectral_contrast
"""

"""
#spectral flatness
#Spectral flatness (or tonality coefficient) is a measure to quantify how much noise-like a sound is, as opposed to being tone-like 

spectral_flatness = []

for aI, audio_excerpt in enumerate(audio_excerpts):
    
    print("aI ", aI, " out of ", audio_excerpts.shape[0])

    flatness = librosa.feature.spectral_flatness(y=audio_excerpt)
    flatness = flatness.flatten()
    
    spectral_flatness.append(flatness)

spectral_flatness = np.stack(spectral_flatness, axis=0)

audio_features["spectral flatness"] = spectral_flatness
"""

"""
#spectral rolloff

spectral_rolloff = []

for aI, audio_excerpt in enumerate(audio_excerpts):
    
    print("aI ", aI, " out of ", audio_excerpts.shape[0])

    rolloff = librosa.feature.spectral_rolloff(y=audio_excerpt, sr=audio_sample_rate)
    rolloff = rolloff.flatten()
    
    spectral_rolloff.append(rolloff)

spectral_rolloff = np.stack(spectral_rolloff, axis=0)

audio_features["spectral rolloff"] = spectral_rolloff
"""

"""
#tempo
# Estimate the tempo (beats per minute)

tempo = []

for aI, audio_excerpt in enumerate(audio_excerpts):
    
    print("aI ", aI, " out of ", audio_excerpts.shape[0])

    tempo_ = librosa.feature.tempo(y=audio_excerpt, sr=audio_sample_rate)
    tempo_ = tempo_.flatten()
    
    tempo.append(tempo_)

tempo = np.stack(tempo, axis=0)

audio_features["tempo"] = tempo
"""


#tempogram
# Compute the tempogram: local autocorrelation of the onset strength envelope. 

tempogram = []

for aI, audio_excerpt in enumerate(audio_excerpts):
    
    print("aI ", aI, " out of ", audio_excerpts.shape[0])

    tempogram_ = librosa.feature.tempogram(y=audio_excerpt, sr=audio_sample_rate)
    tempogram_ = tempogram_.flatten()
    
    tempogram.append(tempogram_)

tempogram = np.stack(tempogram, axis=0)

audio_features["tempogram"] = tempogram


"""
#fourier tempogram
# Compute the Fourier tempogram: the short-time Fourier transform of the onset strength envelope.

fourier_tempogram = []

for aI, audio_excerpt in enumerate(audio_excerpts):
    
    print("aI ", aI, " out of ", audio_excerpts.shape[0])

    tempogram = librosa.feature.fourier_tempogram(y=audio_excerpt, sr=audio_sample_rate)
    tempogram = tempogram.flatten()
    
    fourier_tempogram.append(tempogram)

fourier_tempogram = np.stack(fourier_tempogram, axis=0)
fourier_tempogram = np.stack([np.real(fourier_tempogram), np.imag(fourier_tempogram)], axis=-1)
fourier_tempogram = fourier_tempogram.reshape(audio_excerpts.shape[0], -1)

audio_features["fourier tempogram"] = fourier_tempogram
"""

"""
# tempogram ratio
# Tempogram ratio features, also known as spectral rhythm patterns

tempogram_ratio = []

for aI, audio_excerpt in enumerate(audio_excerpts):
    
    print("aI ", aI, " out of ", audio_excerpts.shape[0])

    ratio = librosa.feature.tempogram_ratio(y=audio_excerpt, sr=audio_sample_rate)
    ratio = ratio.flatten()
    
    tempogram_ratio.append(ratio)

tempogram_ratio = np.stack(tempogram_ratio, axis=0)

audio_features["tempogram ratio"] = tempogram_ratio
"""

"""
KMeans Clustering
"""


from sklearn.cluster import KMeans

cluster_count = 10
random_state = 170

km = KMeans(n_clusters=cluster_count, n_init= "auto", random_state = random_state)
#labels =  km.fit_predict(audio_features["waveform"])
#labels =  km.fit_predict(audio_features["root mean square"])
#labels =  km.fit_predict(audio_features["chroma stft"])
#labels =  km.fit_predict(audio_features["chroma cqt"])
#labels =  km.fit_predict(audio_features["chroma vqt"])
#labels =  km.fit_predict(audio_features["mel spectrogram"])
#labels =  km.fit_predict(audio_features["mfcc"])
#labels =  km.fit_predict(audio_features["spectral centroid"])
#labels =  km.fit_predict(audio_features["spectral bandwidth"])
#labels =  km.fit_predict(audio_features["spectral contrast"])
#labels =  km.fit_predict(audio_features["spectral flatness"])
#labels =  km.fit_predict(audio_features["spectral rolloff"])
#labels =  km.fit_predict(audio_features["tempo"])
labels =  km.fit_predict(audio_features["tempogram"])
#labels =  km.fit_predict(audio_features["fourier tempogram"])
#labels =  km.fit_predict(audio_features["tempogram ratio"])

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

overlap_sc = audio_excerpt_length_sc - audio_excerpt_offset_sc

env_part1 = np.linspace(0.0 ,1.0, overlap_sc)
env_part2 = np.ones(audio_excerpt_length_sc - 2 * overlap_sc, dtype=np.float32)
env_part3 = np.linspace(1.0 ,0.0, overlap_sc)

audio_window_envelope = np.concatenate((env_part1, env_part2, env_part3))

#plt.plot(range(0, audio_window_envelope.shape[0]), audio_window_envelope)


audio_cluster_insert_indices = np.zeros(cluster_count, dtype=np.int32)

for excerpt_index, label in enumerate(labels):
    
    #print("excerpt_index ", excerpt_index, " label ", label, " audio count ", audio_count_per_cluster[label])
    
    if audio_count_per_cluster[label] > 0:
    
        audio_excerpt = audio_excerpts[excerpt_index]
        audio_cluster_insert_index = audio_cluster_insert_indices[label]
        audio_cluster_waveform = audio_clusters[label]
        
        audio_cluster_waveform[audio_cluster_insert_index:audio_cluster_insert_index+audio_excerpt_length_sc] += audio_excerpt * audio_window_envelope
        
        audio_cluster_insert_indices[label] += audio_excerpt_offset_sc

# save audio_cluster waveforms

for cI in range(cluster_count):
    
    #audio_file_name = "waveform_30ms_cluster_{}.wav".format(cI)
    #audio_file_name = "root_mean_square_30ms_cluster_{}.wav".format(cI)
    #audio_file_name = "chroma_stft_30ms_cluster_{}.wav".format(cI)
    #audio_file_name = "chroma_cqt_30ms_cluster_{}.wav".format(cI)
    #audio_file_name = "chroma_vqt_30ms_cluster_{}.wav".format(cI)
    #audio_file_name = "mel_spectrogram_30ms_cluster_{}.wav".format(cI)
    #audio_file_name = "mfcc_30ms_cluster_{}.wav".format(cI)
    #audio_file_name = "spectral_centroid_30ms_cluster_{}.wav".format(cI)
    #audio_file_name = "spectral_bandwidth_30ms_cluster_{}.wav".format(cI)
    #audio_file_name = "spectral_contrast_30ms_cluster_{}.wav".format(cI)
    #audio_file_name = "spectral_flatness_30ms_cluster_{}.wav".format(cI)
    #audio_file_name = "spectral_rolloff_30ms_cluster_{}.wav".format(cI)
    #audio_file_name = "tempo_30ms_cluster_{}.wav".format(cI)
    audio_file_name = "tempogram_1000ms_cluster_{}.wav".format(cI)
    #audio_file_name = "fourier_tempogram_30ms_cluster_{}.wav".format(cI)
    #audio_file_name = "tempogram_ratio_30ms_cluster_{}.wav".format(cI)
    audio_cluster_waveform = audio_clusters[cI]
    
    sf.write(audio_file_name, audio_cluster_waveform, audio_sample_rate)

