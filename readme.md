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
We also provide a scib version with our new function in this repo.


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

The file 'sceval_lib.py' includes all of the metrics we used in this project.

To run the codes in different tasks, please use (we choose batch effect correction as an example here):

```
python sceval_batcheffect.py
```

To avoid using wandb, please set:

```
os.environ["WANDB_MODE"] = "offline"
```

# Results

We have an official website as the summary of our work. Please use this link for access (We will update this link soon, stay tuned!).

# Contact

Please contact tianyu.liu@yale.edu if you have any questions about this project.

# Citation

```
@article {Liu2023.09.08.555192,
	author = {Tianyu Liu and Kexing Li and Yuge Wang and Hongyu Li and Hongyu Zhao},
	title = {Evaluating the Utilities of Large Language Models in Single-cell Data Analysis},
	elocation-id = {2023.09.08.555192},
	year = {2023},
	doi = {10.1101/2023.09.08.555192},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/09/08/2023.09.08.555192},
	eprint = {https://www.biorxiv.org/content/early/2023/09/08/2023.09.08.555192.full.pdf},
	journal = {bioRxiv}
}
```