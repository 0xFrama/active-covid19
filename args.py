import os
import argparse


def inputArguments():
    
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument("--num_classes", default=4, type=int)
    parser.add_argument('--hospitals', nargs='+', type=str, default=['Germania', 'Pavia', 'Lucca', 'Brescia', 'Gemelli - Roma', 'Tione', 'Trento'], 
        help='Name of the hospital / folder to be used.')
    parser.add_argument('--dataset_root', default='preprocessed-58k/', type=str, help='Root folder for the datasets.')
    parser.add_argument('--split_file', default='80_20_activeset.csv', type=str, help='File defining train and test splits.')
    parser.add_argument('--standard_image_size', nargs='+', type=int, default=[250, 250])
    parser.add_argument('--input_image_size', nargs='+', type=int, default=[224, 224])
    parser.add_argument('--domains_count', type=int, default=2)
    parser.add_argument('--domain_label', type=str, default="sensor_label")
    parser.add_argument('--affine_sigma', type=float, default=0.0)
    parser.add_argument('--rotation', type=float, default=23.0)
    # Environment
    parser.add_argument("--epochs", type=int, default=160, help="number of epochs")
    parser.add_argument("--num_workers", default=os.cpu_count() // 2, type=int)
    parser.add_argument('--test_size', default=0.3, type=float, help='Relative size of the test set.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--split', default='patient_hash', type=str, help='The split strategy.')
    parser.add_argument('--stratify', default=None, type=str, help='The field to stratify by.')
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=159, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=159, help="interval evaluations on validation set")
    # Network
    parser.add_argument("--batch_size", default=1, type=int)
    opt = parser.parse_args()

    return opt