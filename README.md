<p align="center">
  <img src ="new_logo.png" alt="drawing" width="800"/>
</p>



An interactive deep-leraning based tool for clustering of genomic sequences.

## Installation via CLI (Linux and macOS)

The installation via command-line interface requires [git](https://git-scm.com/) and [pip](https://pypi.org/project/pip/) installed. After making sure you have them installed on your machine:

0. (**Optional**) Create and activate a new virtual environment
```
$ mkdir iDeLUCS
$ cd iDeLUCS # go to project folder
$ python -m venv myEnv
```

1. Clone this repository
 ```
$ git clone https://github.com/millanp95/iDeLUCS.git .
  ```
2. Install required dependencies
```
$ cd iDeLUCS-master
$ pip install -r requirements.txt 	
```
3. Test installation
```
$ python iDeLUCS.py -h 	
```

## Installation (Windows)
Pending ....


## Usage
iDeLUCS assigns a cluster identifier to all the DNA sequences present in a sigle FASTA file. The path to this file must be provided as input in both the CLI and the GUI versions of iDeLUCS. There are several hyperparameters that are required to perform the clustering. The user may use the default values or select a specific one depending on the amount of information that is available about the dataset. 

### Clustering your own sequences
GUI: 
```
python DeLUCS_GUI.py
```

For command line application: 
```
python iDeLUCS.py <sequence_file>
```

### Clustering parameters

Argument Name| Argument Type | Argument Description | Argument Options
--- | --- | --- | --- 
--n_clusters| integer | Expected or Maximum number of clusters | Default = 5 ; Range: 2-100
--n_epochs| integer |  | Default = 50 ; Recommended Range: 50-150
--n_mimics | integer | | Default = 50 ; Recommended Range: 50-150
--batch_sz | integer | | Default = 3; Recommended Range: 0-600. Note: This value might be limited by the capacity of your machine. 
--GT_file | string | | Default = None.
--k| integer| |  Default = 6, Options: Any integer greater than slice_start and up to the length of the input reference sequence
--optimizer | string | Options: SGD, Adam, RMSprop
--lambda| Float |  | Default = 2.8; Recommended Range  1 (Highly Imbalanced dataset) - 4 (Perfectly Balanced dataset)
--noise |integer | | Default = 1 (keep all non-syn. mutations) ; Recommended Range: 10 -20 

## Example datasets
We provide several datasets to test the perfomance of iDeLUCS to find meaningful genomic signatures for organisms in different kingdoms and at different taxonomic levels.

### Running example datasets
```
python iDeLUCS.py Example/FASTA.fas --GT_file=Example/GT.tsv --n_epochs=50 --lambda=2.8 --k=6 --n_clusters=5 --n_mimics=3 --batch_sz=360
```

Clustering an unkown datasatet with Protists
```
python iDeLUCS.py Example/FASTA_no_labels.fas --n_epochs=30 --lambda=1.2 --k=6 --n_clusters=3 --n_mimics=8 --batch_sz=360
```
