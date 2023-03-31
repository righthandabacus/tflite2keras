"""Command line tool to convert a TFLite model
"""

import argparse
import logging
import os

import tflite
from .model import Model

logger = logging.getLogger('t2k.convert')


def convert(tflite_path: str, keras_path: str, explicit_layouts=None):
    """Converting TensorFlow Lite model (*.tflite) to Keras model.

    Args:
        tflite_path (str): the path to TFLite model.
        keras_path (str): the path where to save the converted Keras model.
        explicit_layouts (dict, optinal): Dict of `str -> tuple(str, str)`.
            For each items, its *tensor name* `->` (*tflite layout*, *keras layout*).
            This can be safely ignored usually - legacy from tflite2onnx
    """

    if not os.path.exists(tflite_path):
        raise ValueError("Invalid TFLite model path (%s)!" % tflite_path)
    if os.path.exists(keras_path):
        logger.warning("Keras model path (%s) existed!", keras_path)

    if explicit_layouts:
        for k, v in explicit_layouts.items():
            if not (isinstance(k, str) and isinstance(v, tuple) and
                    (len(v) == 2) and isinstance(v[0], str) or isinstance(v[1], str)):
                raise ValueError("Invalid explicit layouts!")

    logger.debug("tflite: %s", tflite_path)
    logger.debug("keras: %s", keras_path)
    with open(tflite_path, 'rb') as f:
        buf = f.read()
        im = tflite.Model.GetRootAsModel(buf, 0)

    model = Model(im)
    model.convert(explicit_layouts)
    model.save(keras_path)
    logger.info("Converted Keras model: %s", keras_path)


def cmd_convert():
    from . import __version__, DESCRIPTION
    description = "tflite2keras " + __version__ + ", " + DESCRIPTION
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('tflite_path', help="Path to the input TFLite model")
    parser.add_argument('keras_path', help="Path to save the converted Keras model")

    args = parser.parse_args()

    convert(args.tflite_path, args.keras_path)

if __name__ == "__main__":
    cmd_convert()
