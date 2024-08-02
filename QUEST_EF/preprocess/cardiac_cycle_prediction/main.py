from argparse import ArgumentParser
from pathlib import Path
from QUEST_EF.preprocess.cardiac_cycle_prediction.save_predictions import save_predictions, load_config

def main():
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=Path)
    parser.add_argument('--data_csv', type=Path)

    args = parser.parse_args()
    config_path = args.config_path
    data_csv = args.data_csv
    assert config_path.exists(), f'config_path: {config_path} does not exist'
    model_config_path = 'model_config.yaml'
    model_config = load_config(model_config_path)
    config = load_config(config_path)

    save_predictions(model_config, config, data_csv)

if __name__ == '__main__':
    main()
