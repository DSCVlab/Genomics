# Genomics
The study compares four feature selection  methods and a stacked ensemble implemented in Apache Spark MLliband in single-node scikit-learn.

# Data

The gene-expression data are publicly available and are not redistributed here:
Lim, S.B., et al. *A merged lung cancer transcriptome dataset for clinical predictive modeling.* Scientific Data 5, 180136 (2018). Data set: https://doi.org/10.6084/m9.figshare.5350321.

Analysed data: 1,118 samples (193 normal, 925 NSCLC) and 10,078 genes.

# Environment


OS: UBUNTU 24.04.1 LTS

conda create --name myenv python=3.10
conda activate spark
pip install -r requirements.txt

# Contact

Dr Mohsin Sattar, mohsin08@gmail.com
