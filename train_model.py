#!/usr/bin/env python

import os, sys
import argparse
from base_train import train_12ECG_classifier
from attn_train import train_12ECG_classifier_attn
from intermediate_train import train_12ECG_classifier_intermediate
from irm_train import train_12ECG_classifier_irm
from multilevel_train import train_12ECG_classifier_multilevel
from resnet2_train import train_12ECG_classifier_resnet2
from resnet_train import train_12ECG_classifier_resnet
from siamese_train import train_12ECG_classifier_siamese
from tf2_keras3_refactor import train_12ECG_classifier_tf2
from transformer_train import train_12ECG_classifier_transformer
from baseline_train import train_12ECG_classifier_baseline

def main(args):
    input_directory = args.input_directory
    output_directory = args.output_directory
    load_directory = args.load
    version = args.version

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    print("Running training code...")


    train_12ECG_classifier_resnet2(input_directory, output_directory)

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_directory', type=str, help='Input directory')
    parser.add_argument('output_directory', type=str, help='Output directory')
    parser.add_argument('--load', type=str, help='Model load directory')
    parser.add_argument('--version', type=str, default='base', help='Version of train_12ECG_classifier to use (base, multiclass, attn)')
    args = parser.parse_args()

    main(args)