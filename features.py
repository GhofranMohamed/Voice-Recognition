import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import librosa

def extractWavFeatures(file):

    header = 'chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header = header.split()
    #with file:

    number = file
    y, sr = librosa.load(number, mono=True, duration=30)
    # remove leading and trailing silence
    y, index = librosa.effects.trim(y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=20)
    to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    for e in mfcc:
        to_append += f' {np.mean(e)}'

    # if 'Raouf' in file:
    #     to_append += f' {1}'
    # elif 'arwa' in file:
    #     to_append += f' {2}'
    # elif 'Gufran' in file:
    #     to_append += f' {3}'
    # elif 'Mazen' in file:
    #     to_append += f' {4}'
    # else:
    #     to_append += f' {0}'
    
    return to_append

def extractWavFeaturesDelta(file):

    header = 'chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    for i in range(1, 21):
        header += f' DeltaMfcc{i}'
    header = header.split()
    #with file:

    number = file
    y, sr = librosa.load(number, mono=True, duration=30)
    # remove leading and trailing silence
    y, index = librosa.effects.trim(y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    for e in delta:
        to_append += f' {np.mean(e)}'

    # if 'Raouf' in file:
    #     to_append += f' {1}'
    # elif 'arwa' in file:
    #     to_append += f' {2}'
    # elif 'Gufran' in file:
    #     to_append += f' {3}'
    # elif 'Mazen' in file:
    #     to_append += f' {4}'
    # else:
    #     to_append += f' {0}'
    
    return to_append


def extractWordFeatures(file):

    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    for i in range(1, 21):
        header += f' DeltaMfcc{i}'
    header = header.split()
    y, sr = librosa.load(file, mono=True, duration=30)
    # remove leading and trailing silence
    y, index = librosa.effects.trim(y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    for e in mfcc:
        to_append += f' {np.mean(e)}'
    for e in delta:
        to_append += f' {np.mean(e)}'

    return to_append


# def extractWavFileFeatures(file):
#     # print("The features of the files in the folder "+soundFilesFolder+" will be saved to "+csvFileName)
#     header = 'filename mfcc contrast tonnetz chroma'
#     header += ' label'
#     # header += ' word'
#     header = header.split()

    
#     y, sr = librosa.load(file)
#     # remove leading and trailing silence

#     mfcc = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=40)
#     contrast = librosa.feature.spectral_contrast(y=y,sr=sr)
#     tonnetz = librosa.feature.tonnetz(y=y,sr=sr)
#     chroma = librosa.feature.chroma_stft(y=y,sr=sr)
#     to_append = f'{file.split()[0]} {np.mean(mfcc.T)} {np.mean(contrast.T)} {np.mean(tonnetz.T)} {np.mean(chroma.T)} '
    

#     if 'Raouf' in file:
#         to_append += f' {1}'
#     elif 'arwa' in file:
#         to_append += f' {2}'
#     elif 'Gufran' in file:
#         to_append += f' {3}'
#     elif 'Mazen' in file:
#         to_append += f' {4}'
#     else:
#         to_append += f' {0}'

#     # if 'open' in file:
#     #     to_append += f' {1}'
#     # else:
#     #     to_append += f' {0}'

#     return to_append


# def extractWavFileFeatures2(file):
#     # print("The features of the files in the folder "+soundFilesFolder+" will be saved to "+csvFileName)
#     header = 'filename'
#     for i in range(1, 21):
#         header += f' mfcc{i}'
#     header += ' label'
#     # header += ' word'
#     header = header.split()

#     y, sr = librosa.load(file, mono=True, duration=30)
#     # remove leading and trailing silence
#     y, index = librosa.effects.trim(y)
#     mfcc = librosa.feature.mfcc(y=y, sr=sr)
#     to_append = f'{file.split()[0]} '

#     for e in mfcc:
#         to_append += f' {np.mean(e)}'
#     if 'Raouf' in file:
#         to_append += f' {1}'
#     elif 'arwa' in file:
#         to_append += f' {2}'
#     elif 'Gufran' in file:
#         to_append += f' {3}'
#     else:
#         to_append += f' {0}'

#     # if 'open' in file:
#     #     to_append += f' {1}'
#     # else:
#     #     to_append += f' {0}'

#     return to_append
