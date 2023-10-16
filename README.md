# MR-DNA
This repository includes the implementation of 'MR-DNA: Flexible DNA 5mC-Methylation-Site Recognization in DNA sequences using Token Classification'. 
# Installation
Download MR-DNA from the github repository.

    git clone https://github.com/husonlab/MR-DNA.git
    cd MR-DNA

We recommand you to run MR-DNA in a python vitual environemnt that built by Anaconda, creating conda enviroment equipped with required packages from MR-DNA yml file.

    conda env create -n MR-DNA --file MR-DNA.yml
    conda activate MR-DNA
# Get started
Training DNA-MR on DNA-MR-50 dataset
    
    cd MR-DNA
    python ./script/main.py --dataset MR-DNA-50 --status train --model DistilBertCRF_MethyLoss --savePath ./result/model/

Evaulate DNA-MR performance on test dataset

    cd MR-DNA
    python ./script/main.py --dataset MR-DNA-50 --status test --model DistilBertCRF_MethyLoss

