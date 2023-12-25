# Self-Supervised Learing for Few-Shot Bird Sound Classification
Authors: Ilyass Moummad, Romain Serizel, Nicolas Farrugia
---
We train a feature extractor on the training split of BirdCLEF2020 using self-supervised learning and evaluate it on the validation/test splits following [MetaAudio](https://github.com/CHeggan/MetaAudio-A-Few-Shot-Audio-Classification-Benchmark) Benchmark 

## Data Prepration
Put CNN14 PANN checkpoint "Cnn14_map=0.431.pth" in ```util/``` folder \
pann_selection.py: from the dataset stored in ```--datapath```, this script creates .pt files with the highest PANN activation of birds in ```---targetpath```

For all the training/evaluation scripts, specify ```--datapath``` for the stored data: PANN Selection for ps, and Temporal Proximity for tp

## Training
train_ps.py: train on the PANN selected segments using ```--loss``` loss function \
train_tp.py: train on the whole training set using temporal proximity for sampling two views if ```--tprox``` otherwise it uses two random crops from each audio file \

## Evaluation
eval_ps.py: evalute the model trained using ```--loss``` on the ```--split``` split on the segments selected using PANN \
eval_tp.py: evalute the model trained using ```--loss``` on the ```--split``` split on each chunk of the file, predictions are aggregated using mean \

args.py: contains all the arguments beside the one to be specified above, set to default values, as well as their description \
