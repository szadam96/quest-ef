# QUEST-EF

## Introduction 
The code is structured as follows:
- `preprocess`: Contains the code for preprocessing the data.
- `pretraining`: Contains the code for pretraining the network.
- `training`: Contains the code for training the network.
- `prediction`: Contains the code for predicting the labels of the videos.
- `api`: Contains the code for the REST API to interact with the model.

## Requirements

The code is tested on Python 3.9 and PyTorch 2.0. The full list of dependencies is available in `requirements.txt`.

```bash
pip install -r requirements.txt
pip install .
```

## Weights

The weights of the model used in the paper are available in the `weights` directory in git lfs.

## Datasets

The dataset used for the preptraining in the paper are not publicly available. However, the code can be used with other datasets, with 2D cardiac ultrasound videos that are in a standard DICOM format. The dataset for the suoervised training is available at https://rvenet.github.io/dataset/.

## Preprocess

The preprocess step is used to preprocess the data. It generates the data needed for the pretraining and training steps.

### Usage

```
python preprocess.py (--path_to_data PATH_TO_DATA | --path_to_csv PATH_TO_CSV) --config config/preprocess.yaml --output_dir OUTPUT_DIR --output_csv OUTPUT_CSV
```

The configuration file `config/preprocess.yaml` contains all the parameters for the preprocess step.
The `--path_to_data` argument is used to specify the path to the directory containing the videos. Alternatively a csv file containing the path of each video can be provided with the `--path_to_csv` flag. The `--output_dir` argument is used to specify the path to the directory where the preprocessed data will be saved. The `--output_csv` argument is used to specify the path to the csv file containing the preprocessed data. The table needs to inclide a column named `dicom_path`. In addition, a unique label and orientation can also be provided in the csv for each video. The name of this label must be provided in the config file.

## Pretraining

The pretraining step is used to pretrain the network. It generates the model that will be used for the training step.

### Usage

```bash
python train_self_supervised.py --config config/pretraining.yaml --data_path PATH_TO_DATA --log_path logs/
```

The configuration file `config/pretraining.yaml` contains all the parameters for the pretraining step. The `--data_path` argument is used to specify the path to the directory containing the preprocessed data. The `--log_path` argument is used to specify the path to the directory where the logs will be saved.

## Training

The training step is used to train the network.

### Usage

```bash
python train_supervised.py --config config/training.yaml --data_path PATH_TO_DATA --log_path logs/
```

The configuration file `config/training.yaml` contains all the parameters for the training step. The `--data_path` argument is used to specify the path to the directory containing the preprocessed data. The `--log_path` argument is used to specify the path to the directory where the logs will be saved.

## Prediction

The prediction step is used to predict the labels of the videos.

### Usage

```bash
python predict.py --model_path PATH_TO_MODEL --root_dir PATH_TO_DATA --data_json PATH_TO_JSON --output_dir OUTPUT_DIR --accelerator [gpu|cpu]
```

The `--model_path` argument is used to specify the path to the model.
The `--root_dir` argument is used to specify the path to the directory containing the preprocessed data.
The `--data_json` argument is used to specify the path to the json file containing the data.
The `--output_dir` argument is used to specify the path to the directory where the predictions will be saved.
The `--accelerator` argument is used to specify the device to use for the prediction.

## API

This repo provides an API to interact with the model. The API is built using FastAPI and can be run using the following command:

```bash
fastapi run api/model_prediction.py
```

The API provides one endpoint `/predict` that accepts a POST request, that contains the following parameters:
- `file`: The file containing the video to predict.
- `stanford`: A boolean indicating if the video is in the Stanford orientation.

The API returns a JSON object containing the predictions.