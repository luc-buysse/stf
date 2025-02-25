# ALICE: Adapt your Learnable Image Compression modEl for variable bitrates

  

Pytorch implementation of the paper "ALICE: Adapt your Learnable Image Compression modEl for variable bitrates". This repository is based on [Devil is in the details](https://github.com/InterDigitalInc/CompressAI). The major changes are provided in `compressai/models/stf.py`.

  
  

## Installation

  

Install [CompressAI](https://github.com/InterDigitalInc/CompressAI) and the packages required for development.

```bash

conda  create  -n  compress  python=3.7

conda  activate  compress

pip  install  compressai

pip  install  pybind11

git  clone  https://github.com/luc-buysse/stf.git  stf

cd  stf

pip  install  -e  .

pip  install  -e  '.[dev]'

```

  

## Training

To train adapters for the model, it is necessary to have a SLURM environment available.

The folder stf/configs contain several example configurations of trainings. A valid configuration should respect the yaml formatting and contain the following blocks:
```yaml
model: # A description of the model
	encoder / decoder:
		unfreeze: true # To unfreeze all the parameters of the model
		b1 / b2 / b3 / b4: # For the 4 sequences of transformers
			t1 / t2 / t3 / t4 / t5 / t6: [[alpha: int, rank: int], [alpha: int, rank: int]] # Defines the rank and alpha values of the LoRAs within the MLP layers of the transformers
			a1 / a2 / a3 / a4 / a5 / a6: [[alpha: int, rank: int], [alpha: int, rank: int]]
		symmetrical: true # Only for the decoder, automatically copies the configuration of the encoder

training:
	epochs: int # Number of epochs
	lr: float # Learning rate
	lambda: float # Lambda value for the RD tradeoff
	scheduler: string # By default CosineAnnealing, can also be ReduceLROnPlateau(reduction_factor: float, patience: int)
	original: string # Original model
	dataset: string # Location of the fiftyone dataset folder
	save: string # Name of the file in which the model should be saved

monitor:
	type: tensorboard / wandb
	name: string # Identifier of the training
```

To start a training from a configuration file, navigate to the *stf* folder and execute the command `python3 run.py <config_name>` the name of the configuration should not include the .yml extension.
  
  

### Evaluation

  

To evaluate a trained model on your own dataset, the evaluation script is:

  

```bash

CUDA_VISIBLE_DEVICES=0  python  -m  compressai.utils.eval_model  -d  /path/to/image/folder/  -r  /path/to/reconstruction/folder/  -a  stf  -p  /path/to/checkpoint/  --cuda

```
  
  

### Dataset

The script for downloading [OpenImages](https://github.com/openimages) is provided in `downloader_openimages.py`. Please install [fiftyone](https://github.com/voxel51/fiftyone) first.

  

## Related links

* Devil is in the details: https://github.com/Googolxx/STF.git

* Swin-Transformer: https://github.com/microsoft/Swin-Transformer

* Tensorflow compression library by Ballé et al.: https://github.com/tensorflow/compression

* Range Asymmetric Numeral System code from Fabian 'ryg' Giesen: https://github.com/rygorous/ryg_rans

* Kodak Images Dataset: http://r0k.us/graphics/kodak/

* Open Images Dataset: https://github.com/openimages

* fiftyone: https://github.com/voxel51/fiftyone

* CLIC: https://www.compression.cc/