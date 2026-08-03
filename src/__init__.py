from importlib.metadata import metadata

_meta_ = metadata("tsa-genai")
__version__ = _meta_["Version"]
__author__ = _meta_["Author"]
