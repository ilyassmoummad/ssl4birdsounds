import torch
from torch import nn
from torch.utils.data import DataLoader
from torchaudio import transforms as T
from dataset_ps import BirdClef2020, MEAN, STD
from augmentations import MixRandom, SpecAugment, TimeShift
from losses import SupConLoss, FrobeniusLoss, BarlowTwinsLoss
from mobilenetv3 import mobilenetv3, Projector
from util import Normalization, Standardization
from torchinfo import summary
from args import args
from tqdm import tqdm
import os
        
def train(encoder, projector, train_loader, train_transform, loss_fn, optim, args):

    print(f"Training starting on {args.device}")

    num_epochs = args.epochs

    encoder = encoder.to(args.device)
    encoder.train()
    
    for epoch in range(1, num_epochs+1):
        tr_loss = 0.
        print("Epoch {}".format(epoch))

        for x, y in tqdm(train_loader):

            optim.zero_grad()
            
            x = x.to(args.device)
            y = y.to(args.device)

            x1 = train_transform(x); x2 = train_transform(x)

            z1 = encoder(x1); z2 = encoder(x2)

            h1 = projector(z1); h2 = projector(z2)

            if args.loss == 'supcon':
                loss = loss_fn(h1, h2, y)
            else:
                loss = loss_fn(h1, h2) 

            tr_loss += loss.item()

            loss.backward()
            optim.step()

        tr_loss = tr_loss/len(train_loader)
        print('Average train loss: {}'.format(tr_loss))

    last_state_dict = encoder.state_dict()
    return last_state_dict

if __name__ == "__main__":

    # Dataloader
    train_birdclef = BirdClef2020(datapath=args.datapath, split='train', notpruned=args.notpruned)
    print(f"birdclef train size : {len(train_birdclef)}")
    train_loader = DataLoader(dataset=train_birdclef, batch_size=args.bs, num_workers=args.workers, persistent_workers=True, pin_memory=True, shuffle=True, drop_last=True)

    # Transformations
    time_steps = 251 # int(args.sr*args.duration/args.hoplen)=250
    melspec = T.MelSpectrogram(sample_rate=args.sr, n_fft=args.nfft, hop_length=args.hoplen, f_min=args.fmin, f_max=args.fmax, n_mels=args.nmels).to(args.device)
    power_to_db = T.AmplitudeToDB()
    stft = nn.Sequential(melspec, power_to_db)
    norm = Normalization()
    sd = Standardization(mean=MEAN, std=STD) 
    mix = MixRandom(min_coef=args.mincoef)
    tshift = TimeShift(Tshift=time_steps)
    specaug = SpecAugment(freq_mask=args.fmask, time_mask=args.tmask, freq_stripes=args.fstripe, time_stripes=args.tstripe)
    
    train_transform = nn.Sequential(stft, norm, tshift, mix, specaug, sd)

    # Prepare model
    encoder = mobilenetv3().to(args.device)

    projector = Projector().to(args.device)
    print(summary(encoder))

    # Loss function and optimizer
    print(f"training {args.loss}")
    if args.loss == 'fro':
        loss_fn = FrobeniusLoss(d=args.out_dim, lambd=args.lambd)
    elif args.loss in ['simclr', 'supcon']:
        loss_fn = SupConLoss(temperature=args.tau, device=args.device)
    elif args.loss == 'bt':
        loss_fn = BarlowTwinsLoss(lambda_param=args.lambd, device=args.device)
    optim = torch.optim.AdamW(list(encoder.parameters())+list(projector.parameters()), lr=args.lr, weight_decay=args.wd)

    # Training
    last_state_dict = train(encoder, projector, train_loader, train_transform, loss_fn, optim, args)

    # Saving CKPT
    last_model_path = os.path.join(args.modelpath, args.loss + '.pth')
    torch.save({'encoder': last_state_dict}, last_model_path)
