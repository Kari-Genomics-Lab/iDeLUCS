<p align="center">
  <img src ="new_logo.png" alt="drawing" width="800"/>
</p>



Source code for iDeLUCS. An interactive deep-leraning based tool for clustering of genomic sequences

# Installation

1. Clone this repository
 ```
git clone 
  ```
2. Install required dependencies
```
pip install -r requirements.txt 	
```
3. Test installation
```
python iDeLUCS.py -h 	
```

# Clustering your own sequences
GUI: 
```
python DeLUCS_GUI.py
```

For command line application: 
```
python iDeLUCS.py <sequence_file> --GT_file --n_epochs --lambda --k --n_clusters --n_mimics --batch_sz
```

# Examples
Clustering the Vertebrates dataset
```
python iDeLUCS.py Example/FASTA.fas --GT_file=Example/GT.tsv --n_epochs=30 --lambda=2.8 --k=6 --n_clusters=5 --n_mimics=3 --batch_sz=360
```

Clustering an unkown datasatet with Protists
```
python iDeLUCS.py Example/FASTA_no_labels.fas --n_epochs=30 --lambda=2.8 --k=6 --n_clusters=5 --n_mimics=3 --batch_sz=360
```