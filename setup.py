import numpy as np
from distutils.core import setup
from Cython.Build import cythonize
import sys
sys.path.insert(0, '../')

print
setup(ext_modules = cythonize("bbox_overlaps.pyx"),include_dirs=[np.get_include()])
setup(ext_modules = cythonize("bbox_transform.pyx"),include_dirs=[np.get_include()])
setup(ext_modules = cythonize("cpu_nms.pyx"),include_dirs=[np.get_include()])


