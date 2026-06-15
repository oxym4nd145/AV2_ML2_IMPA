# AV2_ML2_IMPA
Final assignment for the ML2 subject offered at IMPA/IMPA Tech, ministered by Dr. Francisco Ganacim and Dr. Daniel Yukimura

Group composed by:

- [Bruno Pereira de Paula](https://github.com/PereiraFall12)
- [Igor Augusto Zwirtes](https://github.com/igor-zwirtes)
- [Lucas Paulo Gonçalves](https://github.com/oxym4nd145)

---

## The work presented here

The focus of this work is the PINO architecture introduced in the paper [Physics-Informed Neural Operator for Learning Partial Differential Equations](https://arxiv.org/abs/2111.03794), by Zongyi Li et al. 

The main objective of this work is better understand the modern methods for solving PDEs numerically, by:

* Implementing the PINO architecture focusing mostly on direct PyTorch methods;
* Solving simple/popular PDE families using the implemented architecture;
* Visualizing the results in comprehensible/interpretable manners; and
* Comparing the results with other textbook results obtained with other architectures.

The data used in this work comes from the GitHub repository [The Well](https://github.com/polymathicai/the_well). Brief instructions on how to obtain the data from the repository and how to integrate it into PyTorch-friendly formats are presented both below and in the main Python notebook.

---

## How to run this in your machine
### Basic setup
Before anything, you must clone this repository onto your machine. After opening your terminal and accessing the directory you wish to clone this repository into, run:

`git clone https://github.com/oxym4nd145/AV2_ML2_IMPA`

After this, we recommend you set up a Python virtual environment by running these lines of code in your machine's terminal:

`python3 -m venv .env`
`source .env/bin/activate`

After setting up the virtual environment, you have to install the libraries required for running the code in the main notebook. We have these libraries explicit in the file requirements.txt, so you can install them by running this code in your machine's terminal:

`pip install -r requirements.txt`

These steps should be enough for you to navigate cleanly through the main notebooks presented in this work. If you wish to download the data used in this work, follow these next steps. Be warned: the data used in this work is several gigabytes big.

### Downloading the data

As mentioned above, we obtained the data for this work from the [The Well](https://github.com/polymathicai/the_well) repository. With the The Well package installed (should be if you installed the packages in requirements.txt), you can run the command `the-well-download` on your terminal, with the options:

- base-path: WHERE you want to download your data
- dataset: WHAT dataset you want to download
- split: WHAT split of the dataset you want to download

After defining these factors, you may run:

`the-well-download --base-path (your base path) --dataset (dataset name in the The Well repo) --split (split which you want to download (train or test))`

If you skip the `split` option, both splits (train and test) will be downloaded for your dataset. If you skip the --base-path option, the dataset will be downloaded wherever your terminal is accessing. BE WARNED: if you skip the `dataset` option, you will download ALL the datasets in the The Well repo (15tb of data).
