import torch
from torchaudio import transforms as T
from torch.utils.data import Dataset
from utils import Normalization
from tqdm import tqdm
import numpy as np
import glob
import os

MEAN, STD = 0.4977, 0.1205 # computed below if this file is executed

FILES_TO_PRUNE = []
with open(r'util/files_to_prune.txt', 'r') as f:
    for line in f:
        x = line[:-1]
        FILES_TO_PRUNE.append(x)

class BirdClef2020(Dataset):
    def __init__(self, datapath, split, notpruned=False):
        super().__init__()
        
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
        for audiofile in glob.glob(os.path.join(datapath, '*/*.pt')): 
            audiofilename = audiofile.split('/')[-1]
            if not notpruned:
                if audiofilename in FILES_TO_PRUNE:
                    continue
            if audiofile.split('/')[-2].replace(" ","_").replace("'","") in self.split_classes:
                self.audiofiles.append(audiofile)
                cls_label = audiofile.split('/')[-2].replace(" ","_").replace("'","")
                self.audiolabels.append(cls_label)
                self.label.append(self.map_cls_to_int[cls_label])
        self.label = torch.tensor(self.label)

    def __getitem__(self, idx):
        file_path = self.audiofiles[idx]
        data_dict = torch.load(file_path, map_location='cpu')
        wav, label = data_dict['data'], data_dict['label']
        #label = file_path.split('/')[-1]
        #label = self.audiolabels[idx]
        label = self.label[idx]
        return wav, label

    def __len__(self):
        return len(self.audiofiles)

if __name__ == "__main__":

    """ Compute MEAN and STD for the training """

    from args import args

    birdclef = BirdClef2020(datapath=args.datapath, split='train', notpruned=args.notpruned)
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
        wav, label = batch
        wav, label = wav.to(args.device), label.to(args.device)
        mel = stft(wav)
        mel_n = norm(mel)
        mel_n = mel_n.reshape(args.bs, mel_n.size(1), -1)
        mean += mel_n.mean(2).sum(0)
        std += mel_n.std(2).sum(0)
        nb_samples += wav.size(0)

    mean /= nb_samples
    std /= nb_samples

    print(mean, std)
    #0.4977, 0.1205