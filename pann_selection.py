from util.panns import Cnn14
from args import args
import torchaudio
import torch
import glob
import math
import os

root_dir = args.datapath
target_dir = args.targetpath
panns_ckpt = 'pann/Cnn14_mAP=0.431.pth'
cls_folders = [folder for folder in glob.glob(os.path.join(root_dir, '*'))]
DEFAULT_DURATION = args.duration
DEFAULT_SR = args.sr
PANNS_SR = 32000
default_samples = DEFAULT_SR * DEFAULT_DURATION

pann = Cnn14(32000, 1024, 320, 64, 50, 14000, 527)
weights = torch.load(panns_ckpt, map_location='cpu')['model']
state_dict = {k: v for k, v in weights.items() if k in pann.state_dict().keys()}
pann.load_state_dict(state_dict, strict=False)
pann = pann.cuda()
pann.eval()

wavs=[]
strlabels=[]
for cls_folder in cls_folders:
    label = cls_folder.split('/')[-1]
    print(f"creating files for {label}")
    os.makedirs(os.path.join(target_dir, str(label)), exist_ok=True)
    for file in glob.glob(os.path.join(cls_folder, '*.wav')):
        filename = file.split('/')[-1]
        wav, sr = torchaudio.load(file)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav_samples = wav.shape[-1]
        wav_duration = wav_samples / sr
        if wav_duration < float(DEFAULT_DURATION):
            resample = torchaudio.transforms.Resample(sr, DEFAULT_SR)
            wav = resample(wav)
            new_samples = wav.shape[-1]
            ratio = math.ceil(default_samples/new_samples)
            wav = wav.repeat(1, ratio)
            wav = wav[..., :default_samples]
        elif wav_duration > float(DEFAULT_DURATION):
            resample = torchaudio.transforms.Resample(sr, PANNS_SR)
            wav = resample(wav)
            new_samples = wav.shape[-1]
            residual = new_samples % (PANNS_SR * DEFAULT_DURATION)
            if residual != 0:
                wav = wav[..., :new_samples - residual]
            chunk_size = PANNS_SR * DEFAULT_DURATION
            wav_list = torch.split(wav, chunk_size, dim=-1)
            top_activation = 0.
            top_idx = None
            for j, wav_chunk in enumerate(wav_list):
                with torch.no_grad():
                    output_dict = pann(wav_chunk.cuda())
                output = output_dict['clipwise_output'].squeeze().cpu()
                bird_activation = output[111] #bird activation
                if bird_activation > top_activation:
                    top_activation = bird_activation
                    top_idx = j
            resample = torchaudio.transforms.Resample(PANNS_SR, DEFAULT_SR)
            wav = resample(wav_list[top_idx])
            wav = wav.cpu()
        else:
            resample = torchaudio.transforms.Resample(sr, DEFAULT_SR)
            wav = resample(wav)
        filename = filename.replace('wav', 'pt')    
        torch.save({'data': wav.cpu(), 'label': label}, os.path.join(target_dir, str(label), str(filename)))
