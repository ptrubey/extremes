from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
import scipy

# setup(
#     ext_modules = cythonize('cSlice.pyx', annotate = True, language_level = 3),
#     include_dirs = [numpy.get_include()],
#     )
setup(
    ext_modules = cythonize('cProjgamma.pyx', annotate = True, language_level = 3),
    include_dirs = [numpy.get_include()],
    )
setup(
    ext_modules = cythonize('cUtility.pyx', annotate = True, language_level = 3),
    include_dirs = [numpy.get_include()],
    )
# setup(
#     ext_modules = cythonize('hypercube_deviance.pyx', annotate = True, language_level = 3),
#     include_dirs = [numpy.get_include()],
#     extra_compile_args = ['/openmp'],
#     extra_link_args = ['/openmp'],
#     )
ext_modules = [
    Extension(
        'hypercube_deviance',
        ['hypercube_deviance.pyx'],
        include_dirs = [numpy.get_include()],
        extra_compile_args = ['/openmp'],
        extra_link_args = ['/openmp'],
        )
    ]
setup(name = 'hypercube_deviance', ext_modules = cythonize(ext_modules))





# EOF
