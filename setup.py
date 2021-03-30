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
    include_dirs = [numpy.get_include(), scipy.get_include()],
    )
setup(
    ext_modules = cythonize('cUtility.pyx', annotate = True, language_level = 3),
    include_dirs = [numpy.get_include()],
    )
# setup(
#     ext_modules = cythonize('cEnergy.pyx', annotate = True, language_level = 3),
#     include_dirs = [numpy.get_include()],
#     )
# EOF
# setup(
#     ext_modules = cythoize('hypercube_deviance.pyx', annotate = True, language_level = 3),
#     include_dirs = [numpy.get_include()],
#     extra_compile_args = ['/openmp'],
#     extra_link_args = ['/openmp']
# )

# ext_cProjgamma = Extension(
#     'cProjgamma',
#     ['cProjgamma.pyx'],
#     include_dirs = [numpy.get_include(), scipy.get_include()],
#     )
# ext_cUtility = Extension(
#     'cUtility',
#     ['cUtility.pyx'],
#     include_dirs = [numpy.get_include()],
#     )
ext_hypercube_visC = [Extension(
    'hypercube_deviance',
    ['hypercube_deviance.pyx'],
    include_dirs = [numpy.get_include()],
    extra_compile_args = ['/openmp'],
    extra_link_args = ['/openmp'],
    )]
ext_hypercube_gcc = [Extension(
    'hypercube_deviance',
    ['hypercube_deviance.pyx'],
    include_dirs = [numpy.get_include()],
    extra_compile_args = ['-fopenmp'],
    extra_link_args = ['-fopenmp'],
    )]
try:
    setup(
        ext_modules = cythonize('hypercube_deviance.pyx', annotate = True, language_level = 3),
        include_dirs = [numpy.get_include()],
        extra_compile_args = ['/openmp'],
        extra_link_args = ['/openmp'],
        )
except:
    setup(
        ext_modules = cythonize('hypercube_deviance.pyx', annotate = True, language_level = 3),
        include_dirs = [numpy.get_include()],
        extra_compile_args = ['-fopenmp'],
        extra_link_args = ['-fopenmp'],
        )
# setup(name = 'cProjgamma', ext_modules = ext_cProjgamma)
# setup(name = 'cUtility', ext_modules = ext_cUtility)

# try:
#     setup(name = 'hypercube_deviance', ext_modules = ext_hypercube_visC)
# except:
#     setup(name = 'hypercube_distance', ext_modules = ext_hypercube_gcc)






# EOF
