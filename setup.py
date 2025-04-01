from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension

extensions = [
    Extension(
        "gcsa.fovc",
        ["gcsa/fovc.pyx"],
        libraries=["geos_c"],
    ),
]


setup(
    packages=find_packages(),
    ext_modules=cythonize(extensions, annotate=True),
)
