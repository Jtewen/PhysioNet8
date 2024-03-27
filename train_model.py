#!/usr/bin/env python

import os, sys
import argparse
from base_train import train_12ECG_classifier
from multiclass_train import train_12ECG_classifier_multiclass
from attn_train import train_12ECG_classifier_attn
from tf2_keras3_refactor import train_12ECG_classifier as train_12ECG_classifier_refactor

def main(args):
    input_directory = args.input_directory
    output_directory = args.output_directory
    version = args.version

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    print("Running training code...")

    if version == 'base':
        print("Training base model...")
        train_12ECG_classifier(input_directory, output_directory)
    elif version == 'multiclass':
        print("Training multiclass model...")
        train_12ECG_classifier_multiclass(input_directory, output_directory)
    elif version == 'attn':
        print("Training attention model...")
        train_12ECG_classifier_attn(input_directory, output_directory)
    elif version == 'refactor':
        print("Training refactor model...")
        train_12ECG_classifier_refactor(input_directory, output_directory)
    else:
        print("No version flag detected, training base model...")
        train_12ECG_classifier(input_directory, output_directory)

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_directory', type=str, help='Input directory')
    parser.add_argument('output_directory', type=str, help='Output directory')
    parser.add_argument('--version', type=str, default='base', help='Version of train_12ECG_classifier to use (base, multiclass, attn)')
    args = parser.parse_args()

    main(args)