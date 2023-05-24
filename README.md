# MR-DNA
This repository includes the implementation of 'MR-DNA: Flexible DNA 5mC-Methylation-Site Recognization in DNA sequences using Token Classification'. 
# Installation
Download MR-DNA from the github repository.

    git clone https://github.com/husonlab/MR-DNA.git
    cd MR-DNA

We recommand you to run MR-DNA in a python vitual environemnt that built by Anaconda, creating conda enviroment equipped with required packages from MR-DNA yml file.

    conda env create -n MR-DNA --file MR-DNA.yml
    conda activate MR-DNA
# Usage
## Training DNA-MR on DNA-MR-50 dataset
    
    cd scripts
    python distilbert_crf_trainer.py
