from argparse import ArgumentParser
from QUEST_EF.preprocess.view_and_orientation.run_predictions import run_classifier

from QUEST_EF.preprocess.view_and_orientation.utils.config import load_config


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--data_csv', type=str, default=None)

    args = parser.parse_args()

    config = load_config(args.config)
    data_csv = args.data_csv
    run_classifier(config, data_csv, type_='view')
    run_classifier(config, data_csv, type_='orientation')

if __name__ == '__main__':
    main()