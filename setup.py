from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize('cSlice.pyx', annotate = True, language_level = 3),
    include_dirs = [numpy.get_include()],
    )
setup(
    ext_modules = cythonize('cProjgamma.pyx', annotate = True, language_level = 3),
    include_dirs = [numpy.get_include()],
    )
setup(
    ext_modules = cythonize('cUtility.pyx', annotate = True, language_level = 3),
    include_dirs = [numpy.get_include()],
    )
setup(
    ext_modules = cythonize('cEnergy.pyx', annotate = True, language_level = 3),
    include_dirs = [numpy.get_include()],
    )
# EOF
