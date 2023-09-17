# run this test by:
# git clone https://github.com/Redbeard-himalaya/dl-for-emo-tts.git
# cd dl-for-emo-tts
# nvidia-docker run --rm -it --name dl_emtts -h dl_emotts -v ~/redbeard/work/miles_speak/data:/data -v $(pwd)/test.py:/app/test.py dl_emotts:latest bash
# python test.py

import os
import sys
import numpy as np
sys.path.append('pytorch-dc-tts/')
sys.path.append('pytorch-dc-tts/models')
sys.path.append("tacotron_pytorch/")
sys.path.append("tacotron_pytorch/lib/tacotron")


# For the DC-TTS
import torch
from text2mel import Text2Mel
from ssrn import SSRN
from audio import save_to_wav, spectrogram2wav
from utils import get_last_checkpoint_file_name, load_checkpoint_test, save_to_png, load_checkpoint
from datasets.emovdb import vocab, get_test_data

# For the Tacotron
from text import text_to_sequence, symbols
# from util import audio

from tacotron_pytorch import Tacotron
from synthesis import tts as _tts

import warnings
warnings.filterwarnings('ignore')

torch.set_grad_enabled(False)
text2mel = Text2Mel(vocab).eval()

ssrn = SSRN().eval()
load_checkpoint('tacotron_pytorch/trained_models/ssrn.pth', ssrn, None)

model = Tacotron(n_vocab=len(symbols),
                 embedding_dim=256,
                 mel_dim=80,
                 linear_dim=1025,
                 r=5,
                 padding_idx=None,
                 use_memory_mask=False,
)

fs = 20000 #20000
hop_length = 250
model.decoder.max_decoder_steps = 200


def tts_dctts(text2mel, ssrn, text):
    sentences = [text]

    max_N = len(text)
    L = torch.from_numpy(get_test_data(sentences, max_N))
    zeros = torch.from_numpy(np.zeros((1, 80, 1), np.float32))
    Y = zeros
    A = None

    for t in range(210):
        _, Y_t, A = text2mel(L, Y, monotonic_attention=True)
        Y = torch.cat((zeros, Y_t), -1)
        _, attention = torch.max(A[0, :, -1], 0)
        attention = attention.item()
        if L[0, attention] == vocab.index('E'):  # EOS
            break

    _, Z = ssrn(Y)
    Y = Y.cpu().detach().numpy()
    A = A.cpu().detach().numpy()
    Z = Z.cpu().detach().numpy()

    return spectrogram2wav(Z[0, :, :].T), A[0, :, :], Y[0, :, :]


def tts_tacotron(model, text):
    waveform, alignment, spectrogram = _tts(model, text)
    return waveform, alignment, spectrogram 


Emotion = "angry" #@param ["neutral", "angry", "sleepiness", "amused"]
Text = 'The car must be destoryed.' #@param {type:"string"}

wav, align, mel = None, None, None
load_checkpoint_test('tacotron_pytorch/trained_models/'+Emotion.lower()+'_dctts.pth', text2mel, None)


wav, align, mel = tts_dctts(text2mel, ssrn, Text)

from scipy.io.wavfile import write

# vim +61 pytorch-dc-tts/audio.py:
# librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")
# librosa.istft(spectrogram, hop_length=hp.hop_length, win_length=hp.win_length, window="hann")
# vim +47 pytorch-dc-tts/audio.py
# est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
# est = librosa.stft(X_t, n_fft=hp.n_fft, hop_length=hp.hop_length, win_length=hp.win_length)
write('/data/dl_emotts/output.wav', 16000, wav)

