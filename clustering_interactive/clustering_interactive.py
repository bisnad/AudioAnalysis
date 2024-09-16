import audio_analysis
import audio_model
import audio_synthesis
import audio_gui
import audio_control

import librosa
import numpy as np
import os, sys, time, subprocess

"""
Audio Settings
"""

audio_file_path = "D:/Data/audio/Gutenberg/Night_and_Day_by_Virginia_Woolf_48khz.wav"
audio_sample_rate = 48000
audio_excerpt_length = 100 # in milisecs
audio_excerpt_offset = 90 # in milisecs

"""
Load Audio Waveform
"""

audio_waveform, _ = librosa.load(audio_file_path, sr=audio_sample_rate)

audio_waveform = audio_waveform[:audio_sample_rate * 60]
audio_waveform_sc = audio_waveform.shape[0]

"""
Create Audio Excerpts
"""

audio_excerpts = []

audio_excerpt_length_sc = int(audio_excerpt_length / 1000 * audio_sample_rate)
audio_excerpt_offset_sc = int(audio_excerpt_offset / 1000 * audio_sample_rate)

for sI in range(0, audio_waveform_sc - audio_excerpt_length_sc, audio_excerpt_offset_sc):
    
    audio_excerpt = audio_waveform[sI:sI + audio_excerpt_length_sc]
    audio_excerpts.append(audio_excerpt)
    
audio_excerpts = np.stack(audio_excerpts, axis=0)

"""
Calculate Audio Features
"""

audio_features = {}
audio_features["waveform"] = audio_excerpts
audio_features["root mean square"] = audio_analysis.rms(audio_excerpts)
audio_features["chroma stft"] = audio_analysis.chroma_stft(audio_excerpts, audio_sample_rate)
#audio_features["chroma cqt"] = audio_analysis.chroma_cqt(audio_excerpts, audio_sample_rate)
#audio_features["chroma cens"] = audio_analysis.chroma_cens(audio_excerpts, audio_sample_rate)
#audio_features["chroma vqt"] = audio_analysis.chroma_vqt(audio_excerpts, audio_sample_rate)
audio_features["mel spectrogram"] = audio_analysis.mel_spectrogram(audio_excerpts, audio_sample_rate)
audio_features["mfcc"] = audio_analysis.mfcc(audio_excerpts, audio_sample_rate)
audio_features["spectral centroid"] = audio_analysis.spectral_centroid(audio_excerpts, audio_sample_rate)
audio_features["spectral bandwidth"] = audio_analysis.spectral_bandwidth(audio_excerpts, audio_sample_rate)
audio_features["spectral contrast"] = audio_analysis.spectral_contrast(audio_excerpts, audio_sample_rate)
audio_features["spectral flatness"] = audio_analysis.spectral_flatness(audio_excerpts)
audio_features["spectral rolloff"] = audio_analysis.spectral_rolloff(audio_excerpts, audio_sample_rate)
#audio_features["tempo"] = audio_analysis.tempo(audio_excerpts, audio_sample_rate)
audio_features["tempogram"] = audio_analysis.tempogram(audio_excerpts, audio_sample_rate)
#audio_features["tempogram ratio"] = audio_analysis.tempogram_ratio(audio_excerpts, audio_sample_rate)


"""
Load Model
"""

audio_model.config = {
    "audio_excerpts": audio_excerpts,
    "audio_features": audio_features,
    "cluster_method": "kmeans",
    "cluster_count": 20,
    "cluster_random_state": 170
    }

clustering = audio_model.createModel(audio_model.config) 


"""
Setup Audio Synthesis
"""

audio_synthesis.config = {
    "model": clustering,
    "audio_excerpts": audio_excerpts,
    "audio_sample_rate": audio_sample_rate,
    "audio_excerpt_length": audio_excerpt_length,
    "audio_excerpt_offset": audio_excerpt_offset
    }

synthesis = audio_synthesis.AudioSynthesis(audio_synthesis.config)
synthesis.setClusterLabel(1)

"""
GUI
"""

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pathlib import Path

app = QtWidgets.QApplication(sys.argv)
gui = audio_gui.AudioGui()

# set close event
def closeEvent():
    QtWidgets.QApplication.quit()
app.lastWindowClosed.connect(closeEvent) # myExitHandler is a callable

"""
OSC Control
"""

audio_control.config["synthesis"] = synthesis
audio_control.config["model"] = clustering
audio_control.config["ip"] = "127.0.0.1"
audio_control.config["port"] = 9002

osc_control = audio_control.AudioControl(audio_control.config)

"""
Real-Time Audio
"""

import pyaudio

audioBuffer_size = 2048
audioSampleFormat = pyaudio.paFloat32
audioSampleRate = 48000
audioChannelCount = 1

audio_buffer = np.zeros(audioBuffer_size, dtype=np.float32)

p = pyaudio.PyAudio()

def audio_callback(in_data, frame_count, time_info, status):
    
    global audio_buffer
    
    #print("ddream_callback")

    synthesis.update(audio_buffer)
    
    #print("buffer ", audio_buffer)
    
    bybuffer = audio_buffer.tobytes()

    return (bybuffer, pyaudio.paContinue)

stream = p.open(format=audioSampleFormat,
				channels=audioChannelCount,
				rate=audioSampleRate,
				output=True,
				input=False,
				stream_callback=audio_callback,
				frames_per_buffer=audioBuffer_size)


osc_control.start()
gui.show()
app.exec_()

stream.stop_stream()
osc_control.stop()



