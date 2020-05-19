import numpy
from setuptools import Extension, setup
from Cython.Build import cythonize

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

ext_modules = [
    Extension(
        "cut_finder",
        ["cut_finder.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']
    )
]

setup(
    name='cut_finder',
    ext_modules=cythonize(ext_modules, annotate=True),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
    annotate=True
)
