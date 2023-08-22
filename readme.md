# scEval: A evaluation platform for single-cell LLMs

This is the repo for our benchmarking and analysis project. 

# Install

To install our benchmarking environment, please use conda to create a environment based on this yml file in your own machine:
```
conda env create -n scgpt --file scgpt_bench.yml
```

For other methods we used, please refer their original project website for instruction. We recommend creating different environment for different methods.

These methods include: [tGPT](https://github.com/deeplearningplus/tGPT), [Geneformer](https://huggingface.co/ctheodoris/Geneformer), [scBERT](https://github.com/TencentAILabHealthcare/scBERT), [CellLM](https://github.com/BioFM/OpenBioMed/tree/main), [TOSICA](https://github.com/JackieHanLab/TOSICA/tree/main), [ResPAN](https://github.com/AprilYuge/ResPAN/tree/main), [scDesign3](https://github.com/SONGDONGYUAN1994/scDesign3), [scVI](https://scvi-tools.org/), [Tangram](https://github.com/broadinstitute/Tangram).

We need scIB for evaluation. Please use pip to install it:
```
pip install scib
```

# Pre-training weights

Most of our experiments were finished based on weights under [scGPT_bc](https://drive.google.com/drive/folders/1S9B2QUvBAh_FxUNrWrLfsvsds1thF9ad?usp=share_link). [scGPT_full](https://drive.google.com/drive/folders/1eNdHu45uXDHOF4u0J1sYiBLZYN55yytS?usp=share_link) from scGPT v2 was also used in the batch effect correction evaluation.

Pre-training weights of scBERT can be found in [scBERT](https://github.com/TencentAILabHealthcare/scBERT). Pre-training weights of CellLM can be found in [cellLM](https://github.com/BioFM/OpenBioMed/tree/main). Pre-training weights of Geneformer can be found in [Geneformer](https://huggingface.co/ctheodoris/Geneformer).

# Benchmarking information

Please refer different folders for the codes of scEval and metrics we used to evaluate single-cell LLMs under different tasks. In general, we list the tasks and corresponding metrics here:

| Tasks                                                 | Metrics                                  |
|-------------------------------------------------------|------------------------------------------|
| Batch Effect Correction, Multi-omics Data Integration |
| and Simulation                                        | [scIB](https://github.com/theislab/scib)                                     |
| Cell-type Annotation and Gene Function Prediction     | Accuracy, Precision, Recall and F1 score |
| Imputation                                            | [scIB](https://github.com/theislab/scib), Correlation                        |
| Perturbation Prediction                               | Correlation                              |
| Gene Network Analysis                                 | Jaccard similarity                       |

We also have an official website to summarize the main message from our work. Please refer this link for details.

# Contact

Please contact tianyu.liu@yale.edu if you have any questions about this project.