from setuptools import setup, find_packages

VERSION = '0.0.3' 
DESCRIPTION = 'iDeLUCS Python package'
LONG_DESCRIPTION = 'Unsupervised clustering of sequences using latent representations'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="iDeLUCS", 
        version=VERSION,
        author="Pablo Millana",
        author_email="pmillana@uwaterloo.ca",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            "numpy",
            "torch",
            "cython",
            "matplotlib",
            "pandas",
            "torchvision",
            "scikit-learn",
            "scipy",
            "umap-learn",
            "tqdm"
        ], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python', 'clustering', 'sequences', 'latent representations'],
        py_modules=["iDeLUCS"], 
        classifiers= [
            "Intended Audience :: Education",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)