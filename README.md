# Self-Supervised Learning for Few-Shot Bird Sound Classification
Authors: Ilyass Moummad, Romain Serizel, Nicolas Farrugia
---
## Submitted to ICASSP SASB 2024, a more detailed README will be updated upon acceptance

### Preprint: https://arxiv.org/abs/2312.15824

We train a feature extractor on the training split of BirdCLEF2020 using self-supervised learning and evaluate it on the validation/test splits following [MetaAudio](https://github.com/CHeggan/MetaAudio-A-Few-Shot-Audio-Classification-Benchmark) Benchmark 

To Download BirdCLEF 2020 data, go to [Aircrowd](https://www.aicrowd.com/clef_tasks/22/task_dataset_files?challenge_id=211) and make an account to be able to access LifeCLEF 2020 Bird Challenge ressources. \
Go to Ressources and download the file that contains "Download Links" for Train (Train data from the challenge is split into new train/val/test sets for the few-shot benchmark, we refer to MetaAudio for more details.)

## Data Prepration
Put CNN14 PANN checkpoint "Cnn14_map=0.431.pth" in ```util/``` folder \
```pann_selection.py```: from the dataset stored in ```--datapath```, this script creates .pt files with the highest PANN activation of birds in ```---targetpath```

For all the training/evaluation scripts, specify ```--datapath``` for the stored data: ps stands for PANN Selection, and tp stands for Temporal Proximity

## Training
```train_ps.py```: train on the PANN selected segments using ```--loss``` loss function \
```train_tp.py```: train on the whole training set using temporal proximity for sampling two views if ```--tprox``` otherwise it uses two random crops from each audio file

Supported training losses : Barlow Twins ```bt```, SimCLR ```simclr```, FroSSL ```fro```, and SupCon ```supcon```

## Evaluation
```eval_ps.py```: evalute the model trained using ```--loss``` on the ```--split``` split on the segments selected using PANN \
```eval_tp.py```: evalute the model trained using ```--loss``` on the ```--split``` split on each chunk of the file, predictions are aggregated using mean \

```args.py```: contains all the arguments beside the one to be specified above, set to default values, as well as their description

## To cite this work:
```
@misc{moummad2023selfsupervised,
      title={Self-Supervised Learning for Few-Shot Bird Sound Classification}, 
      author={Ilyass Moummad and Romain Serizel and Nicolas Farrugia},
      year={2023},
      eprint={2312.15824},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
