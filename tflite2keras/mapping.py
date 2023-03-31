from tflite import TensorType


def _inverseDict(d):
    return {v: k for k, v in d.items()}


def _buildIndirectMapping(a, b):
    """Given a maps x->y, b maps y->z, return map of x->z."""
    assert len(a) == len(b)
    assert isinstance(list(b.keys())[0], type(list(a.values())[0]))
    c = dict()
    for x in a.keys():
        y = a[x]
        z = b[y]
        c[x] = z
    return c


DTYPE_TFLITE2NAME = {
    TensorType.BOOL: 'bool',
    TensorType.FLOAT16: 'float16',
    TensorType.FLOAT32: 'float32',
    TensorType.INT16: 'int16',
    TensorType.INT32: 'int32',
    TensorType.INT64: 'int64',
    TensorType.INT8: 'int8',
    TensorType.UINT8: 'uint8',
}

DTYPE_NAME2TFLITE = _inverseDict(DTYPE_TFLITE2NAME)
