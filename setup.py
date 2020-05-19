from setuptools import Extension, setup, find_packages
import numpy
from ExKMC import __version__
import sys

with open("README.md", "r") as fh:
    long_description = fh.read()


if '--cython' in sys.argv:
    from Cython.Build import cythonize
    extensions = [
        Extension(
            "cut_finder",
            ["ExKMC/splitters/cut_finder.pyx"],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp']
        )
    ]
    extensions = cythonize(extensions)
    sys.argv.remove("--cython")
else:
    extensions = [Extension("cut_finder", ["ExKMC/splitters/cut_finder.c"])]


setup(
    name="ExKMC",
    version=__version__,
    author="Nave Frost",
    author_email="navefrost@mail.tau.edu",
    liceanse="MIT",
    description="Expanding Explainable K-Means Clustering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/navefr/ExKMC",
    packages=find_packages(),
    ext_modules=extensions,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_dirs=[numpy.get_include()],
    python_requires='>=3.0',
)