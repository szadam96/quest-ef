from pathlib import Path


def get_config(temp_dir, output_path):
    return {
        'data_dir': output_path,
        'image_size': 256,
        'cardiac_cycle_prediction': {},
        'json_generation': {
            'output_dir': Path(temp_dir) / "jsons",
            'frames_to_sample': 16,
            'label': [],
            'name': ''
        },
        'view': {
            'prediction_threshold': 0.5556141668812448,
            'out_dim': 3,
            'batch_size': 100
        },
        'orientation': {
            'prediction_threshold': 0.5,
            'out_dim': 2,
            'batch_size': 100
        }
    }



def cc_model_config():
    return {
        'model_module': {
            'name': 'Echo2dVideosNetLM',
        
            'model': {
                'name': 'UVT',
                'emb_dim': 256,
                'intermediate_size': 1024,
                'num_hidden_layers': 4,
                'img_per_video': None,
                'SDmode': 'reg',
                'rm_branch': 'EF',
                'reg_ed_only': False
            },
            'loss': {
                'name': 'L1Loss'
            },
            'optimizer': {
                'lr': 0.0002,
                'weight_decay': 0.0001
            }},
        'data_module': {
            'name': 'Echo2dPicturesDM',
            'batch_size': 1,
            'num_workers': 1,
        
            'datasets': {
                'predict': {
                    'name': 'Echo2dVideo',
                    'frames_per_video': None,
                    'pictures_dataset': {
                        'name': 'Echo2dPicturesPredict'
                    }
                }
            }
        }
    }