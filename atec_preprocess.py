# coding=utf-8

import sklearn
import tensorflow as tf


def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
        reader = f.readlines()
        lines = []
        for line in reader:
            line = line.strip().split('\t')
            lines.append(line)
        return lines


_read_tsv('data/atec/all.csv')
