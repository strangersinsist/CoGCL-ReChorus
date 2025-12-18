from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')
]

# Import classes for direct access
from .BPRMF import BPRMF
from .BUIR import BUIR
from .CFKG import CFKG
from .DirectAU import DirectAU
from .LightGCN import LightGCN
from .NeuMF import NeuMF
from .POP import POP
