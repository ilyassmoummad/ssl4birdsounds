import torch
import torchaudio
from torchaudio import transforms as T
from torch.utils.data import Dataset
from utils import Normalization
import librosa
import numpy as np
from args import args
from tqdm import tqdm
import random
import math
import glob
import os

MEAN, STD = 0.5550, 0.0770 # computed below if this file is executed

FILES_TO_PRUNE = []
with open(r'util/files_to_prune.txt', 'r') as f:
    for line in f:
        x = line[:-1]
        FILES_TO_PRUNE.append(x)

class BirdClef2020(Dataset):
    def __init__(self, args, datapath, split, notpruned=False):
        super().__init__()

        self.extension = args.ext

        self.duration = args.duration
        self.sr = args.sr
        self.samples = args.sr * args.duration
        
        train_classes, val_classes, test_classes = np.load("util/BirdClef_norm_split_PRUNED.npy", allow_pickle=True)

        train_classes = [train_cls.replace(" ","_").replace("'","") for train_cls in train_classes]
        val_classes = [val_cls.replace(" ","_").replace("'","") for val_cls in val_classes]
        test_classes = [test_cls.replace(" ","_").replace("'","") for test_cls in test_classes]

        self.classes = [cls_folder for cls_folder in glob.glob(os.path.join(datapath, '*/'))]

        self.split_classes = []
        for cls_path in self.classes:
            cls = cls_path.split('/')[-2].replace(" ","_").replace("'","")
            if split == 'train':
                if notpruned:
                    if cls not in (val_classes + test_classes):
                        (self.split_classes).append(cls)
                else:
                    if cls in train_classes:
                        (self.split_classes).append(cls)
            elif split == 'val':
                if cls in val_classes:
                    (self.split_classes).append(cls)
            elif split == 'test':
                if cls in test_classes:
                    (self.split_classes).append(cls)

        self.map_cls_to_int = {}
        for i, cls in enumerate(self.split_classes):
            self.map_cls_to_int[cls] = i

        self.audiofiles = []
        self.audiolabels = []
        self.label = []
        for audiofile in glob.glob(os.path.join(datapath, '*/*.'+self.extension)): 
            audiofilename = audiofile.split('/')[-1]
            if not notpruned:
                if audiofilename.replace(self.extension,'pt') in FILES_TO_PRUNE:
                    continue
            if audiofile.split('/')[-2].replace(" ","_").replace("'","") in self.split_classes:
                self.audiofiles.append(audiofile)
                cls_label = audiofile.split('/')[-2].replace(" ","_").replace("'","")
                self.audiolabels.append(cls_label)
                self.label.append(self.map_cls_to_int[cls_label])
        self.label = torch.tensor(self.label)
        self.deltat = 0
        if args.tprox:
            self.deltat = args.deltat
            self.deltat_samples = args.deltat * args.sr

    def __getitem__(self, idx):
        file_path = self.audiofiles[idx]
        duration = librosa.get_duration(filename=file_path) #filename will be depecrated in librosa future version, and path will replace it
        samplerate = librosa.get_samplerate(path=file_path)
        samples = int(samplerate * duration)
        desired_samples = int(samplerate * self.duration)
        if duration > self.duration + self.deltat:
            if not args.tprox:
                onset1 = random.randint(0, samples-desired_samples)
                onset2 = random.randint(0, samples-desired_samples)

                wav1, sr = torchaudio.load(file_path, onset1, desired_samples)
                wav2, sr = torchaudio.load(file_path, onset2, desired_samples)

                if wav1.shape[0] > 0:
                    wav1 = wav1.mean(dim=0, keepdim=True)
                    wav2 = wav2.mean(dim=0, keepdim=True)
                if sr != args.sr:
                    resample = T.Resample(sr, self.sr)
                    wav1 = resample(wav1)
                    wav2 = resample(wav2)
            else:
                deltat_samples = self.deltat * samplerate
                onset = random.randint(0, samples-desired_samples-deltat_samples)
                wav, sr = torchaudio.load(file_path, onset, desired_samples+deltat_samples)
                if wav.shape[0] > 0:
                    wav = wav.mean(dim=0, keepdim=True)
                if sr != args.sr:
                    resample = T.Resample(sr, self.sr)
                    wav = resample(wav)
                resampled_desired_samples = self.sr * self.duration
                resampled_deltat_samples = self.sr * self.deltat
                wav1 = wav[..., :resampled_desired_samples]
                wav2 = wav[..., resampled_deltat_samples:]
        else:
            wav, sr = torchaudio.load(file_path)
            if sr != args.sr:
                resample = T.Resample(sr, args.sr)
                wav = resample(wav)
            if wav.shape[0] > 0:
                wav = wav.mean(dim=0, keepdim=True)
            wav_len = wav.shape[-1]

            if wav_len > self.samples:
                onset1 = random.randint(0, wav_len - self.samples)
                onset2 = random.randint(0, wav_len - self.samples)
                wav1 = wav[..., onset1:onset1+self.samples]
                wav2 = wav[..., onset2:onset2+self.samples]
            elif wav_len < self.samples:
                ratio = math.ceil(self.samples/wav_len)
                wav = wav.repeat(1, ratio)
                wav = wav[..., :self.samples]
                wav1 = wav2 = wav
            else:
                wav1 = wav2 = wav
        label = self.label[idx]
        return wav1, wav2, label#, file_path

    def __len__(self):
        return len(self.audiofiles)

if __name__ == "__main__":

    """ Compute MEAN and STD for the training """

    from args import args

    birdclef = BirdClef2020(args, datapath=args.datapath, split='train', notpruned=False)
    print(f"trainset size: {len(birdclef)}")

    loader = torch.utils.data.DataLoader(dataset=birdclef, batch_size=args.bs, num_workers=args.workers, drop_last=False)

    # melspec
    melspec = T.MelSpectrogram(sample_rate=args.sr, n_fft=args.nfft, hop_length=args.hoplen, f_min=args.fmin, f_max=args.fmax, n_mels=args.nmels).to(args.device)
    power_to_db = T.AmplitudeToDB()
    stft = torch.nn.Sequential(melspec, power_to_db)
    norm = Normalization()

    mean = 0.
    std = 0.
    nb_samples = 0.
    
    for batch in tqdm(loader):
        wav, _, label = batch
        wav, label = wav.to(args.device), label.to(args.device)
        mel = stft(wav)
        mel_n = norm(mel)
        mel_n = mel_n.reshape(wav.size(0), mel_n.size(1), -1)
        mean += mel_n.mean(2).sum(0)
        std += mel_n.std(2).sum(0)
        nb_samples += wav.size(0)

    mean /= nb_samples
    std /= nb_samples

    print(mean, std)
    #0.5550, 0.0770 random window