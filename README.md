## Setup project
- Setup black, isort, flake and pre-commit by following the instructions in this [link](https://viblo.asia/p/format-code-python-tu-dong-su-dung-isort-black-flake8-va-pre-commit-3P0lPDEolox).

## Commands to setup
We have implemented our work on Python 3.9

Install needed packages
```bash
cd MAC
pip install -r requirements.txt
```

## Prepare datasets
Datasets we used are as follows:
|           Dataset |                                                                            Download |              Comment |
|:-----------------:|:-----------------------------------------------------------------------------------:|----------------------|
| MIMIC-CXR         | [Link](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)                          | official split       |
| CheXpert          | [Link](https://stanfordmlgroup.github.io/competitions/chexpert/)                    | official split for train and val, and `chexpert_5x200` from [GLoRIA](https://stanfordmedicine.app.box.com/s/j5h7q99f3pfi7enc0dom73m4nsm6yzvh) for test |
| VinDr-CXR         | [Link](https://physionet.org/content/vindr-cxr/1.0.0/)                              | official split for test, and random split for train and val |
| RSNA-Pneumonia    | [Link](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data) | same split as [GLoRIA](https://github.com/marshuang80/gloria/blob/416466af1036294301a872e4da169fefc137a192/gloria/datasets/preprocess_datasets.py#L49-L50) |
| SIIM-Pneumothorax | [Link](https://www.kaggle.com/competitions/siim-acr-pneumothorax-segmentation/data) | same split as [GLoRIA](https://github.com/marshuang80/gloria/blob/416466af1036294301a872e4da169fefc137a192/gloria/datasets/preprocess_datasets.py#L90-L91) |

For more details, please refer to [data preparation](./data/README.md).

## Folder structure
```
MAC\
|
├── configs\ # all model configurations go here
├── data\
|   ├── CheXpert
|   |   ├── CheXpert-v1.0\
|   |   |   ├── train\ # train image folder
|   |   |   ├── valid\ # valid image folder
|   ├── imgs\  # folder to mimic-cxr images
|   ├── metadata\ # every metadata file should be inside this folder
|   ├── reports\ # folder to mimic-cxr reports
|   ├── RSNA\
|   |   ├── test\
|   |   ├── train\
|   ├── SIIM\
|   |   ├── test_jpg\
|   |   ├── train_jpg\
|   ├── Vin-CXR\
|   |   ├── data
|   |   |   ├── test_jpg\
|   |   |   ├── train_jpg\
├── data_preparation\ # prompts/scripts to generate/augment texts
├── src\ # folder to source code
├── trainer.py
├── few_shot.py
└── evaluator.py

Run ```trainer.py``` to train model from scratch, here are some parameters use for when running these files:
``` bash
--config: Path to the configuration file. Should be a yaml file
--checkpoint: Path to the checkpoint file to load.
--resume: Boolean flag to resume training from the last checkpoint.
-output-dir: Directory where output files will be saved.
--device: Device to use for training.
--seed: Random seed for reproducibility.
--metadata-path: Path to the metadata CSV file.
```

```train.sh``` provides an example on how to start training.

Run ```few_shot.py``` to fine-tuning on portion of data for few-shot. The parameters are similar to ```trainer.py```. ```few_shot.sh``` provides an example on how to start fine-tuning for few-shot.

```eval.sh``` provides and example on how to evaluating on various datasets.
