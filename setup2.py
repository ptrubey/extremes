# from distutils.core import setup
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize('pointcloud.pyx', annotate = True, language_level = 3),
    include_dirs = [numpy.get_include()],
    )
ext_modules_lcov_visC = [
    Extension(
        'pointcloud',
        ['pointcloud.pyx'],
        extra_compile_args = ['/openmp'],
        extra_link_args = ['/openmp'],
        include_dirs = [numpy.get_include()],
        )
    ]
ext_modules_lcov_gcc = [
    Extension(
        'pointcloud',
        ['pointcloud.pyx'],
        extra_compile_args = ['-fopenmp'],
        extra_link_args = ['-fopenmp'],
        include_dirs = [numpy.get_include()],
        )
    ]
try:
    setup(name = 'pointcloud', ext_modules = ext_modules_lcov_gcc)
except:
    setup(name = 'pointcloud', ext_modules = ext_modules_lcov_visC)

# EOF
