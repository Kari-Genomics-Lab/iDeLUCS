from setuptools import setup, find_packages, Extension

setup(
        include_package_data=True,
        packages=find_packages(),
        
        classifiers= [
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        ext_modules=[Extension(name="idelucs.kmers", sources=["idelucs/kmers.pyx"])]
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
    )
