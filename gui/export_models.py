"""
Export the actor weights of models/*.zip to gui/models/*.npz for the GUI.

Reads the SB3 checkpoint directly (policy.pth is a zip holding a pickle and raw
tensor storages), so torch is not needed. The actor picks moves; the critic (value head) estimates how well the game is going.

    python -m gui.export_models
"""

import collections
import glob
import io
import os
import pickle
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DTYPES = {"FloatStorage": np.float32, "LongStorage": np.int64, "IntStorage": np.int32,
          "HalfStorage": np.float16, "DoubleStorage": np.float64, "BoolStorage": np.bool_}


def load_state_dict(raw):
    z = zipfile.ZipFile(io.BytesIO(raw))
    root = z.namelist()[0].split("/")[0]

    def rebuild(storage, offset, size, stride, *rest):
        size, stride = tuple(size), tuple(stride)
        if not size:
            return storage[offset].copy()
        item = storage.itemsize
        return np.lib.stride_tricks.as_strided(
            storage[offset:], size, [s * item for s in stride]).copy()

    class Unpickler(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2":
                return rebuild
            if name == "_rebuild_parameter":
                return lambda data, *a: data
            if mod == "collections" and name == "OrderedDict":
                return collections.OrderedDict
            if mod.startswith("torch") and name.endswith("Storage"):
                return name
            raise pickle.UnpicklingError(f"Unexpected object {mod}.{name}")

        def persistent_load(self, pid):
            _, stype, key, _, _ = pid
            dtype = DTYPES.get(stype if isinstance(stype, str) else stype.__name__, np.float32)
            return np.frombuffer(z.read(f"{root}/data/{key}"), dtype=dtype)

    return Unpickler(io.BytesIO(z.read(f"{root}/data.pkl"))).load()


def keep(key):
    """Actor and critic weights; the critic answers "how good is this position"
    in the GUI's chat."""
    return not key.endswith("pair_a") and not key.endswith("pair_b")


def main():
    out_dir = os.path.join(ROOT, "gui", "models")
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(ROOT, "models", "*.zip"))):
        sd = load_state_dict(zipfile.ZipFile(path).read("policy.pth"))
        arrays = {k: np.asarray(v, dtype=np.float32) for k, v in sd.items() if keep(k)}
        out = os.path.join(out_dir, os.path.basename(path).replace(".zip", ".npz"))
        np.savez_compressed(out, **arrays)
        print(f"{out}: {sum(a.size for a in arrays.values()):,} parameters")


if __name__ == "__main__":
    main()
