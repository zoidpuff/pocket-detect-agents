"""Restricted unpickler for third-party dataset pickles (LIGYSIS, AF2BIND, ...).

Pickle can execute arbitrary code on load. Only classes from numpy/pandas/collections/datetime
and a few harmless builtins are allowed; anything else raises UnpicklingError, so a malicious or
unexpected pickle fails loudly instead of running code.
"""
import pickle

_ALLOWED_TOP = {"numpy", "pandas", "collections", "datetime"}
_ALLOWED_BUILTINS = {"set", "frozenset", "dict", "list", "tuple", "slice", "complex", "range",
                     "bytearray", "object", "int", "float", "str", "bool"}
_BLOCKED_NAMES = {"load", "loads", "fromfile", "read_pickle", "eval", "system", "exec"}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        top = module.split(".")[0]
        ok = (top in _ALLOWED_TOP and name not in _BLOCKED_NAMES) \
            or (module == "builtins" and name in _ALLOWED_BUILTINS) \
            or (module == "copyreg" and name == "_reconstructor") \
            or (module == "_codecs" and name == "encode")
        if not ok:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")
        return super().find_class(module, name)


def safe_load(path):
    with open(path, "rb") as f:
        return SafeUnpickler(f).load()
