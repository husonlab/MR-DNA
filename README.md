# MR-DNA
This repository includes the implementation of 'MR-DNA: Flexible DNA 5mC-Methylation-Site Recognization in DNA sequences using Token Classification'. 
# Installation
Download MR-DNA from the github repository.

    git clone https://github.com/husonlab/MR-DNA.git
    cd MR-DNA

We recommend you run MR-DNA in a Python virtual environment built by Anaconda, creating a conda environment equipped with the required packages from the MR-DNA yml file.

    conda env create -n MR-DNA --file MR-DNA.yml
    # If failed, please try to update your conda to the latest version using
    conda update -n base -c conda-forge --all
    # Check the installation of the required environment
    conda info --env
    conda activate MR-DNA
# Get started
Training DNA-MR on DNA-MR-50 dataset
    
    cd MR-DNA
    python ./scripts/main.py --dataset MR-DNA-50 --status train --model DistilBertCRF_MethyLoss --savePath ./result/model/

Evaulate DNA-MR performance on test dataset

    cd MR-DNA
    python ./scripts/main.py --dataset MR-DNA-50 --status test --model DistilBertCRF_MethyLoss

