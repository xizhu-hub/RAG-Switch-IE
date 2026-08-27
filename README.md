# RAG-Switch-IE
Core code of the paper Task-Grounded and EL2N-Informed RAG Switch for Domain-Specific Information Extraction

Backbone: LLaMA2-7B  
Dataset: ChemProt
The released code covers the key components required to reproduce and inspect the main experimental pipeline, including training and evaluation of the retrieval decision module. The purpose of this release is to support reproducibility checking during the review process.
Environment
The experiments were conducted with:
```text
Python 3.10.16
```
Below are the key package versions used in this project:
```text
datasets==3.5.0
numpy==2.1.2
pandas==2.2.3
tensorflow==2.19.0
torch==2.5.1+cu121
transformers==4.51.3
```
Please make sure to install the above packages to ensure compatibility and reproducibility.
Depending on your CUDA environment, you may need to install the corresponding PyTorch build from the official PyTorch installation page.
Released Components
This anonymized release provides the core code used for reproducibility verification, including:
EL2N proxy training;
retrieval decision identifier training;
retrieval decision evaluation;
The released code is intended to verify the main methodological components of RAG Switch rather than to serve as a fully packaged production system.
Data
The data setting covered in this release corresponds to:
```text
Backbone: LLaMA2-7B
Dataset: ChemProt
```
The raw ChemProt dataset is not redistributed in this package. Users should obtain the dataset from its official source and follow the input format described in the code/configuration files.
