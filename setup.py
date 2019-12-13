import setuptools
from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy


NAME ='GrasslandModels'
VERSION = '0.1.0'
DESCRIPTION = 'Grassland Productivity Models'
URL = 'https://github.com/sdtaylor/GrasslandModels'
AUTHOR = 'Shawn Taylor'
LICENCE = ''
LONG_DESCRIPTION = """
# GrasslandModels  

## Full documentation  


## Installation

Requires: scipy, pandas, joblib, and numpy

Install from github

```
pip install git+git://github.com/sdtaylor/GrasslandModels
```

## Get in touch

See the [GitHub Repo](https://github.com/sdtaylor/GrasslandModels) to see the 
source code or submit issues and feature requests.

## Acknowledgments

"""

setup(name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description = LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      url=URL,
      author=AUTHOR,
      license=LICENCE,
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      ext_modules = cythonize("GrasslandModels/models/phenograss_cython.pyx"),
      include_dirs = [numpy.get_include()]
)
