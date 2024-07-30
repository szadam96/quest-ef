import argparse
import hashlib
import os
import json
import sys
from pathlib import Path
from matplotlib.pyplot import cla
import numpy as np
import pandas as pd
from cardiac_cycle_prediction.utils.find_cardiac_cycles import es_prediction
from tqdm import tqdm
import warnings
from copy import deepcopy
from typing import Union, List
from create_json import NpEncoder


def calculate_frame_index_arrays(dicom_folder, fps, frames_to_sample):
    es_times = es_prediction(dicom_folder, fps)
    if len(es_times) == 0:
        raise ValueError(f'No cardiac cycles times found for {dicom_folder}')

    results = []

    for es_time in es_times:
        res = np.linspace(es_time[0], es_time[-1], frames_to_sample, dtype=int)
        results.append(res)
    
    return results


def create_data_dict(dicom_labels: pd.DataFrame,
                    path_to_preprocessed_dicoms: Path,
                    frames_to_sample=16,
                    label: Union[str, List[str], None] = ['LVEF', 'RVEF']):
    min_hr = 30
    max_hr = 150

    my_json = dict()
    n_excluded_dicoms = 0
    n_no_cardiac_cycles = 0
    n_valid_dicoms = 0
    n_valid_samples = 0
    excluded_dicoms = []
    for index, row in tqdm(dicom_labels.iterrows(), total=len(dicom_labels), mininterval=0.1):
        dicom_id = row['dicom_id']
        path_to_dicom = path_to_preprocessed_dicoms / dicom_id
        
        if not os.path.exists(path_to_dicom):
            n_excluded_dicoms += 1
            excluded_dicoms.append(path_to_dicom)
            print(f'{path_to_dicom} excluded because the video is not present in the preprocessed data!')
            continue

        frames = [f for f in os.listdir(path_to_dicom/ 'frames')
                         if f.endswith('.png') and os.path.isfile(path_to_dicom / 'frames'/ f)]
        num_of_frames = len(frames)

        if num_of_frames < frames_to_sample:
            n_excluded_dicoms += 1
            excluded_dicoms.append(path_to_dicom)
            print(f'{path_to_dicom} excluded because there are not enough frames in video!')
            continue

        fps = row['fps']
        if pd.isna(row['fps']):
            n_excluded_dicoms += 1
            excluded_dicoms.append(path_to_dicom)
            print(f'{path_to_dicom} excluded because it has invalid fps')
            continue

        try:
            frame_indexes = calculate_frame_index_arrays(path_to_dicom, fps, frames_to_sample)
        except ValueError as e:
            n_no_cardiac_cycles += 1
            print(e)
            continue

        
        my_json[dicom_id] = {
            'frame_indexes': frame_indexes,
            'orientation': row['orientation'],
        }
        
        n_valid_dicoms += 1
        n_valid_samples += len(frame_indexes)
    
    print(f'No. of valid DICOM files: {n_valid_dicoms}')
    print(f'No. of valid cardiac cycles: {n_valid_samples}')
    print(f'No. of excluded DICOM files: {n_excluded_dicoms}')
    print(f'No. of DICOM files with no cardiac cycles found: {n_no_cardiac_cycles}')

    return my_json

def create_jsons(path_to_csv,
                output_path: Path,
                path_to_preprocessed_dicoms: Path,
                name: str = 'multilabel',
                frames_to_sample=None):
    dicom_labels = pd.read_csv(path_to_csv)
    validation_labels = dicom_labels.sample(500, random_state=42)
    train_labels = dicom_labels.drop(validation_labels.index)
    train_dict = create_data_dict(dicom_labels=train_labels,
                                path_to_preprocessed_dicoms=path_to_preprocessed_dicoms,
                                frames_to_sample=frames_to_sample,)
    test_dict = create_data_dict(dicom_labels=validation_labels,
                                path_to_preprocessed_dicoms=path_to_preprocessed_dicoms,
                                frames_to_sample=frames_to_sample)

    os.makedirs(output_path, exist_ok=True)
    with open(output_path / f'train_set_{name}.json', 'w') as outfile:
        json.dump(train_dict, outfile, cls=NpEncoder)
    with open(output_path / f'test_set_{name}.json', 'w') as outfile:
        json.dump(test_dict, outfile, cls=NpEncoder)

    return train_dict, test_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_csv', required=True, help='path to the csv containing the label and uid of the data')
    parser.add_argument('--path_to_preprocessed_data', required=True, help='path to the folder containing the preprocessed videos')
    parser.add_argument('--output_folder', required=True, help='path to the folder where the output jsons will be saved')
    parser.add_argument('--frames_to_sample', type=int, default=None, help="number of frames to use for each video input, if not specified the input type will be set to 'image' instead of video")
    parser.add_argument('--name', default='multilabel', help='name of the jsons')
    parser.add_argument('--external', action='store_true', help='if specified, the jsons will be created for the external set')
    
    args = parser.parse_args()
    path_to_csv = Path(args.path_to_csv)
    path_to_preprocessed_dicoms = Path(args.path_to_preprocessed_data)
    output_folder = Path(args.output_folder)
    frames_to_sample = args.frames_to_sample
    name = args.name

    create_jsons(path_to_csv=path_to_csv,
                 output_path=output_folder,
                 path_to_preprocessed_dicoms=path_to_preprocessed_dicoms,
                 name=name,
                 frames_to_sample=frames_to_sample)

if __name__ == '__main__':
    #athletes()
    main()