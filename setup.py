from setuptools import setup, find_packages, Extension

setup(
        include_package_data=True,
        packages=find_packages(),
        long_description="Deep-learning based tool for clustering genomic sequences",
        
        classifiers= [
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        ext_modules=[Extension(name="idelucs.kmers", sources=["idelucs/kmers.pyx"])]
    )
