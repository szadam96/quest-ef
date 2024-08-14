from fastapi import FastAPI, File, Form, UploadFile, HTTPException
import yaml
from QUEST_EF.preprocess.main import run_view_and_orientation_models
from QUEST_EF.preprocess.save_frames import preprocess_dicom
from QUEST_EF.preprocess.cardiac_cycle_prediction.save_predictions import load_config, save_predictions
from QUEST_EF.preprocess.view_and_orientation.run_predictions import run_classifier
from QUEST_EF.preprocess.create_json import create_jsons
from QUEST_EF.predict import run_prediction
from QUEST_EF.api.utils import get_config, cc_model_config
import pandas as pd
import tempfile
import os
import json
from pathlib import Path

app = FastAPI()

def preprocess_dicom_file(input_path, output_path, flip=False):
    '''
    Preprocess a dicom file

    Parameters
    ----------
    input_path : Path
        Path to the dicom file
    output_path : Path
        Path to the output directory
    flip : bool
        Whether to flip the image
    '''
    res = preprocess_dicom(input_path, output_path, flip=False, raise_error=True)
    if flip:
        res['orientation'] = 'stanford'
    else:
        res['orientation'] = 'mayo'
    pd.DataFrame([res]).to_csv(output_path / "preprocessed.csv", index=False)

def create_data_json(config, output_path):
    '''
    Create a json file for the data

    Parameters
    ----------
    config : dict
        Configuration dictionary
    output_path : Path
        Path to the output directory

    Returns
    -------
    Path
        Path to the json file
    '''
    json_path = config['json_generation']['output_dir']
    frames_to_sample = config['json_generation']['frames_to_sample']
    label = config['json_generation']['label']
    name = config['json_generation']['name']
    print(pd.read_csv(output_path / "preprocessed.csv"))
    create_jsons(path_to_csv=output_path / "preprocessed.csv",
                    output_path=json_path,
                    path_to_preprocessed_dicoms=output_path,
                    frames_to_sample=frames_to_sample,
                    label=label,
                    name=name)
    return json_path / "train_set_.json"
    
def run_ef_prediction(model_checkpoint, root_dir, data_json, output_dir, accerator='cpu'):
    '''
    Run the EF prediction

    Parameters
    ----------
    model_checkpoint : Path
        Path to the model checkpoint
    root_dir : Path
        Path to the root directory
    data_json : Path
        Path to the json file
    output_dir : Path
        Path to the output directory
    accerator : str
        Accelerator to use

    Returns
    -------
    dict
        Dictionary with the LVEF and RVEF predictions
    '''
    run_prediction(model_checkpoint=model_checkpoint,
                    root_dir=root_dir,
                    data_json=data_json,
                    output_dir=output_dir,
                    accerator=accerator)
    
    df = pd.read_csv(output_dir / "validation_predictions.csv")
    view_pred = df['view_pred'].iloc[0]
    if pd.isna(view_pred):
        view_pred = ''
    orientation_pred = df['orientation_pred'].iloc[0]
    if pd.isna(orientation_pred):
        orientation_pred = ''
        
    return {'LVEF': df['lvef_pred'].iloc[0], 'RVEF': df['rvef_pred'].iloc[0],
            'view_pred': view_pred, 'orientation_pred': orientation_pred}

@app.post("/predict/")
async def process_file(file: UploadFile = File(...), stanford: bool = Form(...)):
    '''
    Process a dicom file. The file is preprocessed, the cardiac cycle is predicted, and the EF is predicted.
    Serves as the main endpoint for the API.

    Parameters
    ----------
    file : UploadFile
        The dicom file
    stanford : bool
        Whether the image is in Stanford orientation

    Returns
    -------
    dict
        Dictionary with the LVEF and RVEF predictions
    '''
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir) / file.filename
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())

        output_path = Path(temp_dir) / "preprocessed"
        checkpoints = yaml.safe_load(open("api/checkpoints.yaml", "r"))
        model_config = cc_model_config()
        config = get_config(temp_dir, output_path)
        config['cardiac_cycle_prediction'] = checkpoints['cardiac_cycle_prediction']
        config['view']['checkpoint_path'] = checkpoints['view']['checkpoint_path']
        config['orientation']['checkpoint_path'] = checkpoints['orientation']['checkpoint_path']

        try:
            preprocess_dicom_file(temp_file_path, output_path, flip=stanford)
        except Exception as e:
            if 'File is missing DICOM File Meta Information' in str(e):
                raise HTTPException(status_code=422, detail='The uploaded file is not a valid DICOM file. Please upload another DICOM video')
            else:
                raise HTTPException(status_code=422, detail='Error while processing DICOM file. Please upload another DICOM video')
        
        try:
            run_view_and_orientation_models(config, output_path / "preprocessed.csv", skip_non_a4c=False)
        except Exception as e:
            #raise e
            raise HTTPException(status_code=422, detail=f'Error while processing DICOM file. Please upload another DICOM video')
        df = pd.read_csv(output_path / "preprocessed.csv")
        print(df)
        try:
            save_predictions(model_config=model_config,
                            output_config=config,
                            data_csv=output_path / "preprocessed.csv")
        except Exception as e:
            raise HTTPException(status_code=422, detail=f'Error while processing DICOM file. Please upload another DICOM video')    

        try:
            json_path = create_data_json(config, output_path)
        except Exception as e:
            raise HTTPException(status_code=422, detail='Error while processing DICOM file. Please upload another DICOM video')
        
        try:
            result = run_ef_prediction(model_checkpoint=Path(checkpoints['ef_prediction']['checkpoint_path']),
                          root_dir=output_path,
                            data_json=json_path,
                            output_dir=Path(temp_dir) / "predictions", 
                            accerator=checkpoints['ef_prediction']['accelerator'])
        except Exception as e:
            raise HTTPException(status_code=422, detail=f'Error while processing DICOM file. Please upload another DICOM video')

        return result