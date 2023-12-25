import torch
from torch import nn
import torchaudio
from torchaudio import transforms as T
from torch.utils.data import Dataset, DataLoader, BatchSampler
from utils import Normalization, Standardization
from mobilenetv3 import mobilenetv3
from dataset_tp import MEAN, STD
from torchinfo import summary
import numpy as np
from args import args
import random
import math
import glob
import os

SAMPLES = args.sr * args.duration

class BirdClef2020Eval(Dataset):
    def __init__(self, args, datapath, split):
        super().__init__()

        self.extension = args.ext

        self.duration = args.duration
        self.sr = args.sr
        self.samples = args.sr * args.duration
        
        _, val_classes, test_classes = np.load("util/BirdClef_norm_split_PRUNED.npy", allow_pickle=True)

        val_classes = [val_cls.replace(" ","_").replace("'","") for val_cls in val_classes]
        test_classes = [test_cls.replace(" ","_").replace("'","") for test_cls in test_classes]
 

        self.classes = [cls_folder for cls_folder in glob.glob(os.path.join(datapath, '*/'))]

        self.split_classes = []
        for cls_path in self.classes:
            cls = cls_path.split('/')[-2].replace(" ","_").replace("'","")
            if split == 'val':
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
            if audiofile.split('/')[-2].replace(" ","_").replace("'","") in self.split_classes:
                self.audiofiles.append(audiofile)
                cls_label = audiofile.split('/')[-2].replace(" ","_").replace("'","")
                self.audiolabels.append(cls_label)
                self.label.append(self.map_cls_to_int[cls_label])
        self.label = torch.tensor(self.label)
        self.indices = torch.arange(len(self.label))
    def __getitem__(self, idx):
        file_path = self.audiofiles[idx]
        wav, sr = torchaudio.load(file_path)
        if wav.shape[0] > 0:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != args.sr:
            resample = T.Resample(sr, args.sr)
            wav = resample(wav)    
        label = self.label[idx]
        index = self.indices[idx]
        return wav, label, index

    def __len__(self):
        return len(self.audiofiles)

def nearest_prototype(dataloader, encoder, transform, args):

    print(f"Evaluating on the {args.split} split")

    #encoder.eval()

    accs = []

    for num, (supp_set, query_set) in enumerate(dataloader):

        s_data, s_label = supp_set
        q_data, q_label = query_set

        s_data = s_data.to(args.device)

        with torch.no_grad():
            s_embed = encoder(transform(s_data))

        s_embed = (s_embed) / s_embed.norm(dim=-1, keepdim=True)


        unique_slabels = torch.unique(s_label)

        s_protolabels = []
        s_prototypes = []

        for i, label in enumerate(unique_slabels):
            mask = (s_label == label)
            label_features = s_embed[mask]
            s_prototypes.append(torch.mean(label_features, dim=0, keepdim=True))
            s_protolabels.append(i)

        s_prototypes = torch.stack(s_prototypes)
        s_protolabels = torch.tensor(s_protolabels)

        prototypes_mean = s_prototypes.mean(dim=0)
        s_prototypes -= prototypes_mean

        q_data = q_data.to(args.device)

        with torch.no_grad():
            q_embed = encoder(transform(q_data))

        q_embed = ( q_embed  / q_embed.norm(dim=-1, keepdim=True) ) - prototypes_mean

        unique_qlabels = torch.unique(q_label)
        q_prototypes = []
        q_protolabels = []

        for i, label in enumerate(unique_qlabels):
            mask = (q_label == label)
            label_features = q_embed[mask]
            q_prototypes.append(torch.mean(label_features, dim=0, keepdim=True))
            q_protolabels.append(i)

        q_prototypes = torch.stack(q_prototypes)
        q_protolabels = torch.tensor(q_protolabels)

        distances = torch.cdist(q_prototypes.squeeze(), s_prototypes.squeeze())
        q_predlabels = torch.argmin(distances, dim=-1)

        correct = (q_predlabels.cpu() == q_protolabels).sum()
        print(f"Predicted: {q_predlabels.tolist()} Target: {q_protolabels.tolist()}")
        print(f"number of correct : {correct}")
        total = q_protolabels.size(0)

        acc = correct/total
        accs.append(acc)
        
        print(f"Acc for task n°{num+1} : {acc}")

    accs = torch.tensor(accs)
    ci = 1.96 * ( accs.std() / math.sqrt(args.ntask) )
    print(f"Accuracy : {accs.mean()}[{ci}]({accs.std()})")
    return accs.mean(), accs.std(), ci

def collate_fn(batch):
    wavs, labels, indices = zip(*batch)
    set_size = len(wavs) // 2 
    wavs_s, wavs_q = wavs[:set_size], wavs[set_size:]
    labels_s, labels_q = labels[:set_size], labels[set_size:]
    
    new_wavs_s = []
    new_labels_s = []
    for i, wav in enumerate(wavs_s):
        length = wav.shape[-1]
        nb_label = math.ceil(length/SAMPLES)
        if length > SAMPLES:
            wav_list = list(torch.split(wav, SAMPLES, dim=-1))
            wav_last = wav_list[-1]
            wav_list = wav_list[:-1]
            wav_last_length = wav_last.shape[-1]
            ratio = math.ceil(SAMPLES/wav_last_length)
            wav_last = wav_last.repeat(1, ratio)
            wav_last = wav_last[..., :SAMPLES]
            wav_list.append(wav_last)
            wav_list = torch.stack(wav_list)
            new_wavs_s.append(wav_list)
            for _ in range(nb_label):
                new_labels_s.append(labels_s[i])
        elif length < SAMPLES:
            ratio = math.ceil(SAMPLES/length)
            wav = wav.repeat(1, ratio)
            wav = wav[..., :SAMPLES]
            new_wavs_s.append(wav.unsqueeze(0))
            new_labels_s.append(labels_s[i])
        else:
            new_wavs_s.append(wav.unsqueeze(0))
            new_labels_s.append(labels_s[i])

    new_wavs_s = torch.cat(new_wavs_s, 0)
    new_labels_s = torch.tensor(new_labels_s)

    new_wavs_q = []
    new_labels_q = []
    for i, wav in enumerate(wavs_q):
        length = wav.shape[-1]
        nb_label = math.ceil(length/SAMPLES)
        if length > SAMPLES:
            wav_list = list(torch.split(wav, SAMPLES, dim=-1))
            wav_last = wav_list[-1]
            wav_list = wav_list[:-1]
            wav_last_length = wav_last.shape[-1]
            ratio = math.ceil(SAMPLES/wav_last_length)
            wav_last = wav_last.repeat(1, ratio)
            wav_last = wav_last[..., :SAMPLES]
            wav_list.append(wav_last)
            wav_list = torch.stack(wav_list)
            new_wavs_q.append(wav_list)
            for _ in range(nb_label):
                new_labels_q.append(labels_q[i])
        elif length < SAMPLES:
            ratio = math.ceil(SAMPLES/length)
            wav = wav.repeat(1, ratio)
            wav = wav[..., :SAMPLES]
            new_wavs_q.append(wav.unsqueeze(0))
            new_labels_q.append(labels_q[i])
        else:
            new_wavs_q.append(wav.unsqueeze(0))
            new_labels_q.append(labels_q[i])

    new_wavs_q = torch.cat(new_wavs_q, 0)
    new_labels_q = torch.tensor(new_labels_q)

    return (new_wavs_s, new_labels_s), (new_wavs_q, new_labels_q)

class NWayKShotBatchSampler(BatchSampler):
    def __init__(self, dataset, n_way, k_shot, n_queries, num_batches):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_queries = n_queries
        self.num_batches = num_batches
        self.unique_labels, _ = self.dataset.label.unique(return_counts=True)
        num_classes = len(self.unique_labels)
        self.class_indices = [torch.nonzero(self.dataset.label == class_label).squeeze() for class_label in range(num_classes)]

    def __iter__(self):
        for _ in range(self.num_batches):
            support_set = []
            query_set = []
            selected_labels = random.sample(self.unique_labels.tolist(), self.n_way)

            for label in selected_labels:
                samples_for_label = torch.randperm(len(self.class_indices[label]))[:self.k_shot + self.n_queries]
                support_indices = self.class_indices[label][samples_for_label][:self.k_shot]
                query_indices = self.class_indices[label][samples_for_label][self.k_shot:]

                support_set.extend(support_indices.tolist())
                query_set.extend(query_indices.tolist())

            yield support_set + query_set

    def __len__(self):
        return self.num_batches

if __name__ == "__main__":

    # Dataloader
    eval_birdclef = BirdClef2020Eval(args, datapath=args.datapath, split=args.split)
    print(len(eval_birdclef))

    batch_sampler = NWayKShotBatchSampler(eval_birdclef, args.nway, args.kshot, args.nquery, args.ntask)
    eval_loader = DataLoader(eval_birdclef, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=args.workers, prefetch_factor=args.prefetchfactor)

    # Transformations
    melspec = T.MelSpectrogram(sample_rate=args.sr, n_fft=args.nfft, hop_length=args.hoplen, f_min=args.fmin, f_max=args.fmax, n_mels=args.nmels)
    power_to_db = T.AmplitudeToDB()
    norm = Normalization()
    sd = Standardization(mean=MEAN, std=STD) 
    eval_transform = nn.Sequential(melspec, power_to_db, norm, sd).to(args.device)

    # Prepare model
    encoder = mobilenetv3().to(args.device)
    last_state_dict_path = os.path.join(args.modelpath, args.loss + '.pth')
    last_state_dict = torch.load(last_state_dict_path)
    encoder.load_state_dict(last_state_dict['encoder'])
    print(summary(encoder))

    # Evaluation
    last_test_acc, last_test_std, last_test_ci = nearest_prototype(eval_loader, encoder, eval_transform, args)
    print(f"For {args.ntask} tasks: Mean is {last_test_acc}\tStd is {last_test_std}\tCI is {last_test_ci}")