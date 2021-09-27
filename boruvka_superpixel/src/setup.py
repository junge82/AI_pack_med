# generate cython wrapper for Boruvka Superpixel

from distutils.core import setup, Extension
from Cython.Build import cythonize

import numpy

module = Extension('boruvka_superpixel',
        sources=['boruvka_superpixel_wrap.pyx', 'boruvka_superpixel.cpp'],
        extra_compile_args=['-std=c++11'],
        extra_link_args=['-std=c++11'],
        language='c++',
        )

setup(
        ext_modules =cythonize(module),        
        include_dirs=[numpy.get_include()]
)

# vim: set sw=4 sts=4 expandtab :
