import argparse

parser = argparse.ArgumentParser()

# Generic
parser.add_argument("--device", type=str, default='cuda') #device to train on
parser.add_argument("--datapath", type=str) #birdclef path
parser.add_argument("--targetpath", type=str) #path to store windows selected with PANN
parser.add_argument("--modelpath", type=str) #path to store model weights
parser.add_argument("--ext", type=str, default='wav') #extension of audiofiles
parser.add_argument("--workers", type=int, default=4) #number of workers
parser.add_argument("--sr", type=int, default=16000) #sampling rate
parser.add_argument("--duration", type=int, default=5) #duration in seconds of audios
parser.add_argument("--maxduration", type=int, default=180) #max duration in seconds of audios
parser.add_argument("--notpruned", action='store_true') #keep it False to use birdclef pruned as in MetaAudio (otherwise you use the pruned data for training)

# Data Augmentation
parser.add_argument("--mincoef", type=float, default=0.6) #minimum coef for spectrogram mixing
parser.add_argument("--tprox", action='store_true') #use temporal proximity for two segments
parser.add_argument("--deltat", type=int, default=2) #max seconds between two segments
## SpecAugment
parser.add_argument("--fmask", type=int, default=10) #fmax
parser.add_argument("--tmask", type=int, default=30) #fmax
parser.add_argument("--fstripe", type=int, default=3) #fmax
parser.add_argument("--tstripe", type=int, default=3) #fmax

# Mel Spectrogram
parser.add_argument("--nmels", type=int, default=128) #number of mels
parser.add_argument("--nfft", type=int, default=1024) #size of FFT
parser.add_argument("--hoplen", type=int, default=320) #hop between STFT windows
parser.add_argument("--fmax", type=int, default=8000) #fmax
parser.add_argument("--fmin", type=int, default=50) #fmin

# Loss
parser.add_argument("--loss", type=str, default='bt') #loss to use for training ['fro', 'simclr', 'bt', 'supcon']
parser.add_argument("--tau", type=float, default=1.0) #temperature for cosine sim
parser.add_argument("--lambd", type=float, default=0.01) #loss tradeoff

# Training
parser.add_argument("--bs", type=int, default=256) #batch size for representation learning
parser.add_argument("--epochs", type=int, default=100) #nb of epochs to train the feature extractor on the training set
parser.add_argument("--lr", type=float, default=1e-3) #learning rate 
parser.add_argument("--wd", type=float, default=1e-6) #weight decay

# Few-shot
parser.add_argument("--split", type=str, default='test') #splits to eval on ['val', 'test'] 
parser.add_argument("--nway", type=int, default=5) #number of classes to sample per task
parser.add_argument("--kshot", type=int, default=1) #number of examples to sample per class
parser.add_argument("--nquery", type=int, default=1) #number of queries to sample per class
parser.add_argument("--ntask", type=int, default=1) #number of sampled tasks

args = parser.parse_args()
