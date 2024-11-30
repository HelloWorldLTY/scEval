import os
GPU_NUMBER = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
os.environ["NCCL_DEBUG"] = "INFO"

# imports
import datetime
import subprocess
import math
import matplotlib.pyplot as plt
import pickle
import random
import numpy as np
import pandas as pd
import time
from collections import Counter
from datasets import load_from_disk
from sklearn import preprocessing
import sklearn.metrics
from sklearn.metrics import accuracy_score, auc, confusion_matrix, ConfusionMatrixDisplay, roc_curve, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
import torch
from transformers import BertForTokenClassification
from transformers import Trainer, set_seed 
from transformers.training_args import TrainingArguments
from tqdm.notebook import tqdm

import scanpy as sc 
import anndata as ad

from geneformer import DataCollatorForGeneClassification
from geneformer.pretrainer import token_dictionary
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from geneformer.in_silico_perturber import pad_tensor_list, quant_layers

sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=200, facecolor='white')

# load token dictionary (Ensembl IDs:token)
with open(TOKEN_DICTIONARY_FILE, "rb") as f:
    gene_token_dict = pickle.load(f)

#### Functions for preocessing gene labels

# function for preparing targets and labels
def prep_inputs(genegroup1, genegroup2, id_type):
    if id_type == "gene_name":
        targets1 = [gene_name_id_dict[gene] for gene in genegroup1 if gene_name_id_dict.get(gene) in token_dictionary]
        targets2 = [gene_name_id_dict[gene] for gene in genegroup2 if gene_name_id_dict.get(gene) in token_dictionary]
    elif id_type == "ensembl_id":
        targets1 = [gene for gene in genegroup1 if gene in token_dictionary]
        targets2 = [gene for gene in genegroup2 if gene in token_dictionary]
            
    targets1_id = [token_dictionary[gene] for gene in targets1]
    targets2_id = [token_dictionary[gene] for gene in targets2]
    
    targets = np.array(targets1_id + targets2_id)
    labels = np.array([0]*len(targets1_id) + [1]*len(targets2_id))

    print(f"# targets1: {len(targets1_id)}\n# targets2: {len(targets2_id)}")
    return targets, labels

#### Functions for evaluation

def preprocess_classifier_batch(cell_batch, max_len):
    if max_len == None:
        max_len = max([len(i) for i in cell_batch["input_ids"]])
    def pad_label_example(example):
        example["labels"] = np.pad(example["labels"], 
                                   (0, max_len-len(example["input_ids"])), 
                                   mode='constant', constant_values=-100)
        example["input_ids"] = np.pad(example["input_ids"], 
                                      (0, max_len-len(example["input_ids"])), 
                                      mode='constant', constant_values=token_dictionary.get("<pad>"))
        example["attention_mask"] = (example["input_ids"] != token_dictionary.get("<pad>")).astype(int)
        return example
    padded_batch = cell_batch.map(pad_label_example)
    return padded_batch

# forward batch size is batch size for model inference (e.g. 200)
# used for inference on fine-tuned model
def classifier_predict(model, evalset, forward_batch_size):
    predict_logits = []
    predict_labels = []
    model.eval()
    
    # ensure there is at least 2 examples in each batch to avoid incorrect tensor dims
    evalset_len = len(evalset)
    max_divisible = find_largest_div(evalset_len, forward_batch_size)
    if len(evalset) - max_divisible == 1:
        evalset_len = max_divisible
    
    max_evalset_len = max(evalset.select([i for i in range(evalset_len)])["length"])
    
    for i in range(0, evalset_len, forward_batch_size):
        max_range = min(i+forward_batch_size, evalset_len)
        batch_evalset = evalset.select([i for i in range(i, max_range)])
        padded_batch = preprocess_classifier_batch(batch_evalset, max_evalset_len)
        padded_batch.set_format(type="torch")
        
        input_data_batch = padded_batch["input_ids"]
        attn_msk_batch = padded_batch["attention_mask"]
        label_batch = padded_batch["labels"]
        with torch.no_grad():
            outputs = model(
                input_ids = input_data_batch.to("cuda"), 
                attention_mask = attn_msk_batch.to("cuda"), 
                labels = label_batch.to("cuda"), 
            )
            predict_logits += [torch.squeeze(outputs.logits.to("cpu"))]
            predict_labels += [torch.squeeze(label_batch.to("cpu"))]
            
    logits_by_cell = torch.cat(predict_logits)
    all_logits = logits_by_cell.reshape(-1, logits_by_cell.shape[2])
    
    labels_by_cell = torch.cat(predict_labels)
    all_labels = torch.flatten(labels_by_cell)
    
    logit_label_paired = [item for item in list(zip(all_logits.tolist(), all_labels.tolist())) if item[1]!=-100]
    
    y_pred = [vote(item[0]) for item in logit_label_paired]
    y_true = [item[1] for item in logit_label_paired]
    logits_list = [item[0] for item in logit_label_paired]
    
    # probability of class 1
    y_score = [py_softmax(item)[1] for item in logits_list]
    conf_mat = sklearn.metrics.confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    # calculate metrics
    roc_auc = auc(fpr, tpr)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    macro_precision = precision_score(y_true, y_pred, average='macro')
    weighted_precision = precision_score(y_true, y_pred, average='weighted')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    weighted_recall = recall_score(y_true, y_pred, average='weighted')
    
    metric_dict = {
      'auc': roc_auc,
      'accuracy': acc,
      'macro_f1': macro_f1,
      'weighted_f1': weighted_f1,
      'macro_precision': macro_precision,
      'weighted_precision': weighted_precision,
      'macro_recall': macro_recall,
      'weighted_recall': weighted_recall
    }
    
    # plot roc_curve for this split
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.show()
    # interpolate to graph
#     interp_tpr = np.interp(mean_fpr, fpr, tpr)
#     interp_tpr[0] = 0.0
    
#     return fpr, tpr, interp_tpr, conf_mat
    return metric_dict, fpr, tpr, conf_mat

def vote(logit_pair):
    a, b = logit_pair
    if a > b:
        return 0
    elif b > a:
        return 1
    elif a == b:
        return "tie"
    
def py_softmax(vector):
    e = np.exp(vector)
    return e / e.sum()
    
# get cross-validated mean and sd metrics
# def get_cross_valid_metrics(all_tpr, all_roc_auc, all_tpr_wt):
#     wts = [count/sum(all_tpr_wt) for count in all_tpr_wt]
#     print(f'Weights:\n{wts}')
#     all_weighted_tpr = [a*b for a,b in zip(all_tpr, wts)]
#     mean_tpr = np.sum(all_weighted_tpr, axis=0)
#     mean_tpr[-1] = 1.0
    
#     all_weighted_roc_auc = [a*b for a,b in zip(all_roc_auc, wts)]
#     roc_auc = np.sum(all_weighted_roc_auc)
#     roc_auc_sd = math.sqrt(np.average((all_roc_auc-roc_auc)**2, weights=wts))
#     return mean_tpr, roc_auc, roc_auc_sd

# Function to find the largest number smaller
# than or equal to N that is divisible by k
def find_largest_div(N, K):
    rem = N % K
    if(rem == 0):
        return N
    else:
        return N - rem

#### Functions for splitting training and evaluation sets

# Function for splitting genes into train-gene-sets and eval-gene-sets, then processing dataset to generate train-cell-sets and eval-cell-sets
def train_eval_split(dataset,
                     targets,
                     labels,
                     train_size: float=0.7):
    
    # Split genes into train_set and eval_set
    n_genes = targets.shape[0]
    all_index = [i for i in range(0,n_genes)]
    train_index = random.sample(range(0, n_genes), round(n_genes*train_size))
    train_index.sort()
    eval_index = [i for i in all_index if i not in train_index]
    train_index, eval_index = np.array(train_index), np.array(eval_index)
    
    targets_train, targets_eval = targets[train_index], targets[eval_index]
    labels_train, labels_eval = labels[train_index], labels[eval_index]
    label_dict_train = dict(zip(targets_train, labels_train))
    label_dict_eval = dict(zip(targets_eval, labels_eval))
    
    # label conversion functions
    def generate_train_labels(example):
        example["labels"] = [label_dict_train.get(token_id, -100) for token_id in example["input_ids"]]
        return example

    def generate_eval_labels(example):
        example["labels"] = [label_dict_eval.get(token_id, -100) for token_id in example["input_ids"]]
        return example
        
    # label datasets 
    print(f"Labeling training data")
    trainset_labeled = dataset.map(generate_train_labels)
    print(f"Labeling evaluation data")
    evalset_labeled = dataset.map(generate_eval_labels)
    
    return trainset_labeled, evalset_labeled, label_dict_train, label_dict_eval 

#### Training settings

# set model parameters
# max input size
max_input_size = 2 ** 11  # 2048

# set training hyperparameters
# max learning rate
max_lr = 5e-5
# how many pretrained layers to freeze
freeze_layers = 4
# number gpus
num_gpus = 1
# number cpu cores
num_proc = 8
# batch size for training and eval
geneformer_batch_size = 12
# learning schedule
lr_schedule_fn = "linear"
# warmup steps
warmup_steps = 500
# number of epochs
epochs = 5
# optimizer
optimizer = "adamw"

#### Functions for fine-tune training

trainset, evalset, label_dict_train, label_dict_eval = pickle.load(open('/gpfs/gibbs/pi/zhao/kl945/Geneformer/finetuned_models/230811_geneformer_GeneClassifier_dosageTF_L2048_B12_LR5e-05_LSlinear_WU500_E5_Oadamw_F4_FCFalse/train_eval_splits.p', 'rb'))

# fine-tune and evaluate
def finetune_evaluate(dataset,
                      targets, 
                      labels, 
                      train_size: float=0.7,
                      freeze_layers: int=4,
                      seed: int=42,
                      output_dir=None,
                      filter_cells=False,
                      num_proc=8,
                      train_eval_set_path=None,
                      dataset_name = 'task2'):
    
    # set training arguments
    training_args = {
        "learning_rate": max_lr,
        "do_train": True,
#         "do_eval": False,
        "evaluation_strategy": "no",
        "save_strategy": "epoch",
        "logging_steps": 100,
        "group_by_length": True,
        "length_column_name": "length",
        "disable_tqdm": False,
        "lr_scheduler_type": lr_schedule_fn,
        "warmup_steps": warmup_steps,
        "weight_decay": 0.001,
        "per_device_train_batch_size": geneformer_batch_size,
        "per_device_eval_batch_size": geneformer_batch_size,
        "num_train_epochs": epochs,
#         "load_best_model_at_end": True
        }
    
    # define output directory path
    current_date = datetime.datetime.now()
    datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
    if output_dir==None:
        output_dir = f"/gpfs/gibbs/pi/zhao/tl688/cpsc_finalproject/genept_data/geneformer_code/finetuned_models/{dataset_name}_{datestamp}_geneformer_GeneClassifier_dosageTF_S{seed}_L{max_input_size}_B{geneformer_batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_E{epochs}_O{optimizer}_F{freeze_layers}_FC{filter_cells}/"
    model_dir = os.path.join(output_dir, "models/")
    
    # ensure not overwriting previously saved model
    model_test = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.isfile(model_test) == True:
        raise Exception("Model already saved to this directory.")

    # make output directory
    subprocess.call(f'mkdir {output_dir}', shell=True)
    subprocess.call(f'mkdir {model_dir}', shell=True)
    
    # initiate eval metrics to return
    n_classes = len(set(labels))
#     mean_fpr = np.linspace(0, 1, 100)
    confusion = np.zeros((n_classes, n_classes))
    
    # split labeled genes into train and eval set
    if train_eval_set_path==None:
        trainset, evalset, label_dict_train, label_dict_eval = train_eval_split(dataset=dataset,
                                                                             targets=targets,
                                                                             labels=labels,
                                                                             train_size=train_size)
    else: # use previous splits
        trainset, evalset, label_dict_train, label_dict_eval = pickle.load(open(train_eval_set_path, 'rb'))
    
    # filter cells without genes in labeled set
    if filter_cells:
        # function to filter by whether contains train or eval labels
        def if_contains_train_label(example):
            a = label_dict_train.keys()
            b = example['input_ids']
            return not set(a).isdisjoint(b)

        def if_contains_eval_label(example):
            a = label_dict_eval.keys()
            b = example['input_ids']
            return not set(a).isdisjoint(b)
        
        # filter dataset for examples containing classes for this split
        print(f"Filtering training data")
        trainset = trainset.filter(if_contains_train_label, num_proc=num_proc)
        print(f"Filtered {round((1-len(trainset)/len(dataset))*100)}%; {len(trainset)} remain\n")
        
        print(f"Filtering evalation data")
        evalset = evalset.filter(if_contains_eval_label, num_proc=num_proc)
        print(f"Filtered {round((1-len(evalset)/len(dataset))*100)}%; {len(evalset)} remain\n")
    
    # save train-eval splits
    with open(os.path.join(output_dir, 'train_eval_splits.p'), 'wb') as f:
        pickle.dump((trainset, evalset, label_dict_train, label_dict_eval), f)
        
    # load model
    set_seed(seed=seed)
    model = BertForTokenClassification.from_pretrained(
            "/gpfs/gibbs/pi/zhao/kl945/Geneformer/pretrained_model/",
            num_labels=2,
            output_attentions = False,
            output_hidden_states = False
        )
    if freeze_layers is not None:
        modules_to_freeze = model.bert.encoder.layer[:freeze_layers]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False
                
    model = model.to("cuda:0")
        
    # add output directory to training args and initiate
    training_args["output_dir"] = output_dir
    training_args_init = TrainingArguments(**training_args)
        
    # create the trainer
    trainer = Trainer(
            model=model,
            args=training_args_init,
            data_collator=DataCollatorForGeneClassification(),
            train_dataset=trainset,
            eval_dataset=evalset
        )
    
    # train the gene classifier
    start_time = time.time()
    trainer.train()
    training_mem = torch.cuda.memory_allocated()
    end_time = time.time()
    training_time = end_time - start_time
    print(f'Total memory allocated after training: {training_mem}')
    print(f'Time for fine-tuning training: {training_time}')
        
    # save model
    trainer.save_model(model_dir)
        
    # evaluate model and save metrics
    start_time = time.time()
    metrics, fpr, tpr, conf_mat = classifier_predict(trainer.model, evalset, 200)
    end_time = time.time()
    eval_time = end_time - start_time
    metrics['training_time'] = training_time
    metrics['evaluation_time'] = eval_time
    metrics['training_memory_allocation'] = training_mem 
    
    print(metrics)
    with open(os.path.join(output_dir, 'metrics.p'), 'wb') as f:
        pickle.dump(metrics, f)
    
    return metrics, fpr, tpr, conf_mat

def stability_test(run_name, start_seed=0, end_seed=10):
    df_metrics = pd.DataFrame(columns=['auc','accuracy','macro_f1','weighted_f1','macro_precision','weighted_precision','macro_recall',
                                       'weighted_recall','training_time','evaluation_time','training_memory_allocation']) 
    # Load dataset
    dataset = load_from_disk(f'../data/datasets/{run_name}.dataset/')
    dataset = dataset.shuffle(seed=42)
    # table of corresponding Ensembl IDs, gene names, and gene types (e.g. coding, miRNA, etc.)
    gene_info = pd.read_csv("../examples/example_datasets/gene_classification/gene_info_table.csv", index_col=0)
    # create dictionaries for corresponding attributes
    gene_id_type_dict = dict(zip(gene_info["ensembl_id"],gene_info["gene_type"]))
    gene_name_id_dict = dict(zip(gene_info["gene_name"],gene_info["ensembl_id"]))
    gene_id_name_dict = {v: k for k,v in gene_name_id_dict.items()}
    
    # preparing targets and labels for dosage sensitive vs insensitive TFs
    dosage_tfs = pd.read_csv("../examples/example_datasets/gene_classification/dosage_sens_tf_labels.csv", header=0)
    sensitive = dosage_tfs["dosage_sensitive"].dropna()
    insensitive = dosage_tfs["dosage_insensitive"].dropna()
    targets, labels = prep_inputs(sensitive, insensitive, "ensembl_id")
    
    # Fine-tuning with seeds from 0 to 9
    conf_mat_list = []
    for i in range(start_seed, end_seed):
        print(f'Fine-tuning with seed: {i}')
        print(f'Initial memory allocated: {torch.cuda.memory_allocated()}')
        metrics, fpr, tpr, confusion_matrix = finetune_evaluate(dataset=dataset,
                            targets=targets,
                            labels=labels,
                            train_eval_set_path='./finetuned_models/230811_geneformer_GeneClassifier_dosageTF_L2048_B12_LR5e-05_LSlinear_WU500_E5_Oadamw_F4_FCFalse/train_eval_splits.p',
                            seed=i 
                           )    
        df_metrics.loc[i] = metrics
        conf_mat_list.append(confusion_matrix)
        
    
    return df_metrics

### Fine-tuning on Panglao dataset

# Load dataset
dataset = load_from_disk('/gpfs/gibbs/pi/zhao/tl688/geneformer_class_out/data/datasets/panglao.dataset/')
dataset = dataset.shuffle(seed=42)
dataset

#### Load gene attribute information

# table of corresponding Ensembl IDs, gene names, and gene types (e.g. coding, miRNA, etc.)
gene_info = pd.read_csv("/gpfs/gibbs/pi/zhao/kl945/Geneformer/examples/example_datasets/gene_classification/gene_info_table.csv", index_col=0)

# create dictionaries for corresponding attributes
gene_id_type_dict = dict(zip(gene_info["ensembl_id"],gene_info["gene_type"]))
gene_name_id_dict = dict(zip(gene_info["gene_name"],gene_info["ensembl_id"]))
gene_id_name_dict = {v: k for k,v in gene_name_id_dict.items()}


# preparing targets and labels for dosage sensitive vs insensitive TFs
dosage_tfs = pd.read_csv("/gpfs/gibbs/pi/zhao/kl945/Geneformer/examples/example_datasets/gene_classification/dosage_sens_tf_labels.csv", header=0)
sensitive = dosage_tfs["dosage_sensitive"].dropna()
insensitive = dosage_tfs["dosage_insensitive"].dropna()
targets, labels = prep_inputs(sensitive, insensitive, "ensembl_id")

metrics, fpr, tpr, confusion_matrix = finetune_evaluate(dataset=dataset,
                            targets=targets,
                            labels=labels,
                            train_eval_set_path=None
#                             training_args=training_args,
#                             filter_cells=False,
#                             freeze_layers=freeze_layers
                           )

import resource
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  / (1e6) )
import torch
print(torch.cuda.max_memory_allocated()/1024/1024/1024)

