import torch
from torch import nn
from torch.utils.data import DataLoader, BatchSampler
from torchaudio import transforms as T
from dataset_ps import BirdClef2020, MEAN, STD
from mobilenetv3 import mobilenetv3
from torchinfo import summary
from utils import Normalization, Standardization
from args import args
import random
import math
import os

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

def nearest_prototype(dataloader, encoder, transform, args):

    print(f"Evaluating on the {args.split} split")

    #encoder.eval()

    accs = []

    for ntask in range(1, args.ntask+1):

        for data, label in dataloader:

            s_data, q_data = data[:args.nway * args.kshot, ...], data[args.nway * args.kshot:, ...]
            s_label, q_label = label[:args.nway * args.kshot, ...], label[args.nway * args.kshot:, ...]

            # Compute Support Prototypes

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

            # Compute Query Prototypes

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

            # Prediction

            distances = torch.cdist(q_prototypes.squeeze(), s_prototypes.squeeze())
            q_predlabels = torch.argmin(distances, dim=-1)

            correct = (q_predlabels.cpu() == q_protolabels).sum()
            print(f"Predicted: {q_predlabels.tolist()} Target: {q_protolabels.tolist()}")
            print(f"number of correct : {correct}")
            total = q_protolabels.size(0)

            acc = correct/total
            accs.append(acc)
            
            print(f"Acc for task n°{ntask} : {acc}")

    accs = torch.tensor(accs)
    ci = 1.96 * ( accs.std() / math.sqrt(args.ntask) )
    print(f"Accuracy : {accs.mean()}[{ci}]({accs.std()})")
    return accs.mean(), accs.std(), ci

if __name__ == "__main__":

    # Val
    eval_birdclef = BirdClef2020(datapath=args.datapath, split='val', notpruned=args.notpruned)
    batch_sampler = NWayKShotBatchSampler(eval_birdclef, args.nway, args.kshot, args.nquery, 1)
    eval_loader = DataLoader(eval_birdclef, batch_sampler=batch_sampler)

    # Data transformations
    time_steps = 251 # int(args.sr*args.duration/args.hoplen)=250
    melspec = T.MelSpectrogram(sample_rate=args.sr, n_fft=args.nfft, hop_length=args.hoplen, f_min=args.fmin, f_max=args.fmax, n_mels=args.nmels)
    power_to_db = T.AmplitudeToDB()
    norm = Normalization()
    sd = Standardization(mean=MEAN, std=STD) 
    val_transform = nn.Sequential(melspec, power_to_db, norm, sd).to(args.device)

    # Prepare model
    encoder = mobilenetv3().to(args.device)
    last_state_dict_path = os.path.join(args.modelpath, args.loss + '.pth')
    last_state_dict = torch.load(last_state_dict_path)
    encoder.load_state_dict(last_state_dict['encoder'])
    print(summary(encoder))

    # Evaluation
    last_test_acc, last_test_std, last_test_ci = nearest_prototype(eval_loader, encoder, val_transform, args)
    print(f"For {args.ntask} tasks: Mean is {last_test_acc}\tStd is {last_test_std}\tCI is {last_test_ci}")
