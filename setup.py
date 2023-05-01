from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize

VERSION = '0.0.1' 
DESCRIPTION = 'My first Python package'
LONG_DESCRIPTION = 'My first Python package with a slightly longer description'

import numpy

cython_directives = {
    'embedsignature': True,
}

extensions = cythonize([
    Extension(name='idelucs.kmers',
              sources=["idelucs/kmers.pyx"],
              include_dirs=[numpy.get_include()]),
], compiler_directives=cython_directives)



# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="idelucs", 
        version=VERSION,
        author="Pablo Millan",
        author_email="<pmillana@uwaterloo.ca>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        zip_safe=False,            # Without these two options
        include_package_data=True, # PyInstaller may not find your C-Extensions
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Operating System :: Linux :: Ubuntu",

        ]
)
