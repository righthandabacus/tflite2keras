"""Converting TensorFlow Lite models (*.tflite) to Keras models (*.h5)"""

from .convert import convert
from .utils import enableDebugLog, getSupportedOperators

# package metadata
__name__ = 'tflite2keras'
__version__ = '0.0.1'
DESCRIPTION = "Convert TensorFlow Lite models to Keras"

__all__ = [
    convert,
    enableDebugLog,
    getSupportedOperators,
    __name__,
    __version__,
    DESCRIPTION,
]
