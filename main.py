import argparse
import json
from functools import lru_cache
from math import log
from textwrap import wrap

import librosa as librosa
import matplotlib.pyplot as plt
import numpy as np
from librosa import samples_to_time, frames_to_time
from librosa.feature.spectral import rms
from moviepy import editor
from moviepy.video.io.bindings import mplfig_to_npimage

_PITCH_HOP = 512
_PITCH_FRAMELEN = 1024
_FRAME_TIMESTEP = 0.1  # seconds

# things to save as global variables where caching isn't allowed
PITCH = np.empty((0, 0))

@lru_cache()
def get_duration(wav):
    audio, sr = librosa.load(wav)
    timelabels = samples_to_time(range(len(audio)), sr=sr)
    return timelabels[-1]


# @lru_cache()  # can't cache due to audio datatype
def estimate_pitch_simple(audio, sampling_rate):
    global PITCH
    if not PITCH.size > 0:
        pitch, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            hop_length=_PITCH_HOP,
            frame_length=_PITCH_FRAMELEN,
            sr=sampling_rate)
        PITCH = pitch
    return PITCH


# function to add colorbar for imshow data and axis
def add_colorbar_outside(im, ax, distance=0.01):
    fig = ax.get_figure()
    bbox = ax.get_position()
    width = 0.01
    eps = distance  # margin between plot and colorbar

    cax = fig.add_axes([bbox.x1 + eps, bbox.y0, width, bbox.height])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Intensity dB')
    return fig


def plot_pitch(pitch_vals, pitch_ax, sr, until=True):
    time_labels = frames_to_time(np.arange(len(pitch_vals)),
                                 sr=sr,
                                 n_fft=_PITCH_FRAMELEN,
                                 hop_length=_PITCH_HOP)

    # until defaults to true so all timestamps are included
    excl_index = len(list(filter(lambda x: x <= until, time_labels)))
    pitch_ax.plot(time_labels[:excl_index], pitch_vals[:excl_index],
                  color="lime", linewidth=5)

    # axis settings
    # maximum pitch, excluding NaN
    pitch_ax.set_ylim(0, max(np.where(np.isnan(pitch_vals), 0.0, pitch_vals)) + 100)
    pitch_ax.set_ylabel("F0 (Hz)", color="lime")
    pitch_ax.yaxis.set_label_position("right")
    pitch_ax.tick_params(labelsize="x-small",
                         colors="lime",
                         bottom=False,
                         labelbottom=False,
                         left=False,
                         labelleft=False,
                         right=True,
                         labelright=True)


@lru_cache()
def prep_audio(wav):
    audio, sampling_rate = librosa.load(wav)
    time_labels = samples_to_time(np.arange(len(audio)),
                                  sr=sampling_rate)
    return audio, sampling_rate, time_labels


def estimate_rms_energy(audio):
    S, phase = librosa.magphase(librosa.stft(audio,
                                             n_fft=_PITCH_FRAMELEN,
                                             hop_length=_PITCH_HOP,
                                             # win_length=None,
                                             ))

    return rms(S=S, frame_length=_PITCH_FRAMELEN, hop_length=_PITCH_HOP)[0]


def plot_energy(energy_vals, energy_ax, sr, until=True):
    time_labels = frames_to_time(np.arange(len(energy_vals)),
                                 sr=sr,
                                 n_fft=_PITCH_FRAMELEN,
                                 hop_length=_PITCH_HOP)

    excl_index = len(list(filter(lambda x: x <= until, time_labels)))

    energy_ax.semilogy(time_labels[:excl_index], energy_vals[:excl_index],
                       color='purple',
                       linewidth=5,
                       label='RMS Energy')

    # axis settings
    # maximum pitch, excluding NaN
    energy_ax.set_ylabel('RMS Energy', color='purple')
    energy_ax.yaxis.set_label_coords(0.03, 0.5)
    energy_ax.tick_params(labelsize='x-small',
                          colors='purple',
                          direction='in',
                          pad=-45,
                          bottom=False,
                          labelbottom=False,
                          left=True,
                          labelleft=True,
                          right=False,
                          labelright=False)


def plot_audio_static(wav, fig_title, top='wave', timestamped_transcription=False, pitch=True, energy=True, until=True):
    # load audio
    audio, sampling_rate, time_labels = prep_audio(wav)
    # in case a fig is still open from a previous call to the function
    # adding a cache caused a bug with the vline
    plt.close()

    # figure settings
    plt.rcParams.update({'font.size': 25,
                         'figure.figsize': (30, 20)})
    fig, (ax, ax2) = plt.subplots(nrows=2, sharex='all')
    title = fig.suptitle('\n'.join(wrap(fig_title, 90)))
    title.set_y(0.93)

    # top plot
    # simple waveform
    if top == 'wave':
        excl_index = len(list(filter(lambda x: x <= until, time_labels)))
        ax.plot(time_labels[:excl_index], audio[:excl_index])
        ax.set_ylim(0 - (max(audio) + 0.01), max(audio) + 0.01)
        ax.set_ylabel('Amplitude')
    elif top == 'pitch':
        # TODO find sleeker way to choose which feature gets displayed where
        pass
    elif top == 'energy':
        pass

    # bottom plot
    # spectrogram
    *_, t, im = ax2.specgram(audio, Fs=sampling_rate,
                             noverlap=250, NFFT=300,
                             cmap='afmhot')

    ax2.set_ylim(0, 5000)
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (s)')

    if pitch:
        pitch_vals = estimate_pitch_simple(audio, sampling_rate)
        # overlay bottom plot
        pitch_ax = ax2.twinx()
        plot_pitch(pitch_vals, pitch_ax, sampling_rate, until=until)

    if energy:
        energy_vals = estimate_rms_energy(audio)
        # overlay bottom plot
        energy_ax = ax2.twinx()
        plot_energy(energy_vals, energy_ax, sampling_rate, until=until)
