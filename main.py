import argparse
import json
import os
from functools import lru_cache
from textwrap import wrap

import imageio
import librosa as librosa
import matplotlib.pyplot as plt
import numpy as np
# pip install pygifsicle
import pygifsicle as pygifsicle
import textgrid
from librosa import samples_to_time, frames_to_time
from moviepy import editor
from moviepy.video.io.bindings import mplfig_to_npimage

PITCH_HOP = 512
PITCH_FRAMELEN = 1024
FRAME_TIMESTEP = 0.1  # seconds


def load_textgrid(textgrid_file, tier, as_frames=True):
    tg = textgrid.TextGrid.fromFile(textgrid_file)
    tier = get_tier(tg, tier)
    if as_frames:
        return [(item.mark,
                 librosa.time_to_frames(float(item.minTime), sr=22050, hop_length=256, n_fft=None),
                 librosa.time_to_frames(float(item.maxTime), sr=22050, hop_length=256, n_fft=None))
                for item in tier]
    else:
        return [(item.mark,
                 float(item.minTime),
                 float(item.maxTime))
                for item in tier]


def get_tier(textgrid, tier_name):
    for tier in textgrid.tiers:
        if tier.name == tier_name:
            return tier
    # else None is returned


@lru_cache()
def get_duration(wav):
    audio, sr = librosa.load(wav)
    timelabels = samples_to_time(range(len(audio)), sr=sr)
    return timelabels[-1]


def estimate_pitch_simple(audio, sampling_rate):
    pitch, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        hop_length=PITCH_HOP,
        frame_length=PITCH_FRAMELEN,
        sr=sampling_rate)
    return pitch


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


def plot_audio_static(wav, fig_title, timestamped_transcription=False, pitch=True):
    # load audio
    audio, sampling_rate = librosa.load(wav)

    # figure settings
    plt.rcParams.update({'font.size': 25,
                         'figure.figsize': (30, 20)})
    fig, (ax, ax2) = plt.subplots(nrows=2, sharex='all')
    title = fig.suptitle('\n'.join(wrap(fig_title, 90)))
    title.set_y(0.93)

    # top plot
    time_labels = samples_to_time(np.arange(len(audio)),
                                  sr=sampling_rate)
    # simple waveform
    ax.plot(time_labels, audio)
    ax.set_ylabel('Amplitude')

    # bottom plot
    # spectrogram
    *_, t, im = ax2.specgram(audio, Fs=sampling_rate,
                             noverlap=250, NFFT=300,
                             cmap='afmhot')

    ax2.set_ylim(0, 5000)
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (s)')

    if pitch:
        # overlay bottom plot
        pitch_vals = estimate_pitch_simple(audio, sampling_rate)

        pitch_ax = ax2.twinx()
        time_labels = frames_to_time(np.arange(len(pitch_vals)),
                                     sr=sampling_rate,
                                     n_fft=PITCH_FRAMELEN,
                                     hop_length=PITCH_HOP)
        pitch_ax.plot(time_labels, pitch_vals, color="lime", linewidth=5)

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

    # needs space for F0 axis if present
    fig = add_colorbar_outside(im, ax2, distance=0.04 if pitch else 0.01)

    if timestamped_transcription:
        phone_list = load_textgrid(timestamped_transcription, tier='phones', as_frames=False)
        # add lines to mark transcription boundaries on both plots
        for phone, min_time, max_time in phone_list[:-1]:
            ax.axvline(x=max_time, linewidth=5)
            ax2.axvline(x=max_time, linewidth=5)

        # add transcription symbol per symbol
        for index, (phone, min_time, max_time) in enumerate(phone_list):
            text = phone.strip('}').strip('{').strip('012')
            timestamp_difference = (max_time - min_time)
            timestamp_middle = min_time + (timestamp_difference / 2)

            ax2.text(timestamp_middle, 5500, text, fontsize=36,
                     horizontalalignment='center',
                     verticalalignment='center')

    fig.align_ylabels([ax, ax2])
    # removing excessive whitespace on the left
    # which makes the figure look off-center
    plt.subplots_adjust(left=0.05)
    return fig


def create_gif_with_progress_line(wav, text, video_path, textgrid=None):
    duration = get_duration(wav)

    def single_frame(t):
        figure = plot_audio_static(wav, text, timestamped_transcription=textgrid)
        for fig_ax in figure.axes:
            if fig_ax.get_ylabel().lower().startswith('intensity'):
                continue
            fig_ax.axvline(x=t, linewidth=4, color='maroon')

        return mplfig_to_npimage(figure)

    video = editor.VideoClip(single_frame, duration=duration)
    audio = editor.AudioFileClip(wav)
    final_video = video.set_audio(audio)

    final_video.write_videofile(fps=1/FRAME_TIMESTEP, codec='libx264', filename=video_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav')
    parser.add_argument('--text')
    parser.add_argument('--textgrid', required=False)
    parser.add_argument('--output-video')

    args = parser.parse_args()

    create_gif_with_progress_line(args.wav, args.text, args.output_video, args.textgrid)
