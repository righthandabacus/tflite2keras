tflite2keras: Convert TensorFlow Lite models to Keras
=====================================================

`tflite2keras` is a tool to convert TensorFlow Lite (TFLite) models
(`*.tflite`) to Keras models (`*.h5`).

This is modified from [`tflite2onnx`](https://github.com/zhenhuaw-me/tflite2onnx), depends on
[tflite](https://github.com/zhenhuaw-me/tflite) to read the `*.tflite` files, and
**highly experimental** with a lot of layers not supported.

## Usage

```
pip install tflite
cd tflite2keras   # this git repository
python -m tflite2keras.convert --help
```

## What is this?

TFLite is a set of operators and tensors. Which they are connected in a graph.
Executing a TFLite model is to assign the input tensor(s), invoke the model so
the tensors populate along the graph, and retrieve the output tensors.

Keras is a high-level library that *roughly* match the TFLite operators. They
are similar (e.g., for images, both assumed NHWC order). If the TFLite model is
simple enough, we should be able to extract the layer parameters and
reconstruct the corresponding Keras layer. And because the graph is in TFLite
model, we should also be able to connect the Keras layers to reproduce the
model.

However, not all TFLite operators have corresponding layers and Keras and vice
versa. This tool would not work in those cases.
