# Emergent Ability analysis

Here we discuss out experiment design to analyze emergent ability of single-cell LLMs. All the results and pipelines here are related to Figure 21 in the main text.

# Cross-data cell-type annotation

We compared the performance of scGPT to vanilla NN based on the crossing-data cell type annotation. The datasets here include "demo_train.h5ad" and "demo_test.h5ad". They are from Pancreas. Codes here are related to "Cell type Annotation".

# Cross-specises cell-type annotation

We compared the performance of scGPT to vanilla NN based on cell type prediction for 1. spatial transcriptomics and 2. mouse cell atlas seperated by batch. Codes here are related to "Cell type Annotation".

# Spatial transcriptomics batch effect correction.

We colelct spatial transcriptomics from human brain without cell labels and reduce the batch effect of the two datasets based on scGPT. Codes here are related to "Batch Effect Correction".


