"""Record the runtime used for a reproduction run."""
import platform
from importlib.metadata import version, PackageNotFoundError

def environment():
    names=['torch','transformers','open_clip_torch','numpy','pandas','scipy','scikit-learn','Pillow']
    out={'python':platform.python_version(),'platform':platform.platform()}
    for name in names:
        try:out[name]=version(name)
        except PackageNotFoundError:out[name]=None
    return out
