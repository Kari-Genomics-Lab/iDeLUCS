<p align="center">
  <img src ="itallics_iDeLUCS_logo.png" alt="drawing" width="800"/>
</p>


An interactive deep-learning based tool for clustering genomic sequences.

## Installation for the CLI Application:

The recommended installation process is via terminal and it requires [pip](https://pypi.org/project/pip/) installed. Follow the steps after making sure you have it installed on your machine.

0. (**Optional**) Create and activate a new virtual environment using [venv](https://docs.python.org/3/library/venv.html)
```
python3 -m venv dev_iDeLUCS
source dev_iDeLUCS/bin/activate
```
or if you are on Windows:
```
python3 -m venv dev_iDeLUCS
dev_iDeLUCS/Scripts/activate
```

1. Clone this reposotory
```
git clone https://github.com/millanp95/iDeLUCS.git iDeLUCS
```

2. Install the package

If you do not plan to use the GUI you can also install the package directly from Pypi:
```
pip install ideLUCS
```
Otherwise:
```
cd iDelUCS
python setup.py build_ext --inplace
pip install -e .
```

3. Test installation
```
idelucs -h 	
```

## Installation for the GUI:
No further steps are needed after clonning the repository if you are comfortable with executing the script from the terminal.
```
pip install PyQt5	
```
If you do not want to execute the tool form the terminal, you may download the executable files:

- For Windows: here.
- For MAC OS: here.

**Note:** iDeLUCS uses PyTorch as the machine learning development framework and its latest stable release might not be compatible with your version of CUDA. The GUI is built using the Qt platform, make sure you have Qt platform plugings installed. 


### (Optional) Using the GUI on Apple Silicon-based Macs
The current `pip` distribution of `PyQt5` does not support Apple Silicon. However, to extend the functionality of iDeLCUS GUI to MAC users (Thanks to the works of Shane Ding), it is possible to use the `brew` distribution of `PyQt5` and then point to that distribution from within the `virtualenv`.
1. Ensure that `homebrew` is installed correctly.
2. `brew install PyQt5`
3. Run `brew --prefix PyQt5` to see where `PyQt5` is installed.
4. Append `/lib/python3.9/site-packages/PyQt5` to the end of the string returned from the previous command.
5. Navigate to the root `iDeLUCS` folder.
6. Establish a symbolic link between the `brew` installed `PyQt5` package and the virtual environment's `site-packages` folder.
    - Example: `ln -s /opt/homebrew/opt/pyqt@5/lib/python3.9/site-packages/PyQt5 /Users/shaneding/Desktop/iDeLUCS/dev_iDeLUCS/lib/python3.10/site-packages`

If the above does not work, try seeing what different versions of python are inside `/opt/homebrew/opt/pyqt@5/lib` and try linking to a different version.


## Running iDeLUCS

We provide several datasets to test the perfomance of iDeLUCS to find meaningful genomic signatures for organisms in different kingdoms and at different taxonomic levels. But the process is similar on the user's own data.

Clustering a dataset with grouth truth:
```
idelucs --sequence_file=Example/Vertebrata.fas --GT_file=Example/Vertebrata_GT.tsv --n_epochs=35 --lambda=2.8 --k=6 --n_clusters=5 --n_mimics=3 --batch_sz=512
```

Clustering an unkown dataset:
```
idelucs --sequence_file=Example/Actinopterygii.fas --n_epochs=30 --k=6 --n_clusters=3 --n_mimics=8 --batch_sz=256
```

Running iDeLUCS GUI from terminal: 
```
cd iDeLCUS/GUI
python GUI_iDeLUCS.py
```

### Clustering parameters

iDeLUCS assigns a cluster identifier to all the DNA sequences present in a sigle FASTA file. The path to this file must be provided as input in both the CLI and the GUI versions of iDeLUCS. There are several hyperparameters that are required to perform the clustering. The user may use the default values or select a specific one depending on the amount of information that is available about the dataset. 

Argument Name| Variable Type | Description | Options
--- | --- | --- | --- 
<sequence_file>| string | Path for single Fasta file with training sequences | Required to run the program. 
--GT_file | string | Path for a tab-separated file with possible labels for the training dataset. These labels won't be used during training, just for posthoc analysis. The GT file must contain the columns: `sequence_id` with the sequence identifiers, these identifiers must correspond to the identifiers provided in the `<sequence_file>`. The column `cluster_id` with the "ground truth" assignments for each sequence. Additionally, the user may include an additional column with alternative labeling under the header `phylogeny` | Default = None.
--k| integer|k-mer length|  Default = 6, Options:`[4,5,6]`
--n_clusters| integer|Expected or maximum number of clusters to find. It should be at least `n_true_clusters` when `GT_file` is provided'.Use 0 to launch iDeLCUS + HDBSCAn to automatically find the number of clusters in fine-grained clustering.| Default = 5 ; Range: 0, 2-199.
--n_epochs| integer | Number of training epochs. An epoch is defined as a training iteration over all the training pairs.' | Default: `50` ; Recommended Range: `[50,150]`
--n_mimics | integer | Number of data augmentations per sequence that will be considered during training | Default = 50; Recommended Range: 50-150
--batch_sz | integer | Number of data pairs the network will receive simultaneously during training. A larger batch may improve convergence but it may harm the accuracy. It must be a multiple of `n_mimcs`.| Default: 3; Recommended Range: 0-600. **Note**: This value might be limited by the capacity of your machine. 
--optimizer | string | Optimization algorithm to train the neural network | `['SGD', 'Adam','RMSprop']`
--lambda| float | Hyperparameter to control cluster balance. | Default = 2.8; Recommended Range  1 (highly imbalanced dataset) - 3 (perfectly balanced dataset)
--scheduler | string | Learning rate schedule used to train the neural network | `['None', 'Plateau','Triangle']`


