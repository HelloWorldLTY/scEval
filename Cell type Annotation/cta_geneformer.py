import os
GPU_NUMBER = [0]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
os.environ["NCCL_DEBUG"] = "INFO"

# imports
from collections import Counter, defaultdict
import datetime
import time 
import pickle
import subprocess
import seaborn as sns; sns.set()
from datasets import load_from_disk, Dataset
import numpy as np 
import pandas as pd 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import BertForSequenceClassification, BertForTokenClassification, BertForMaskedLM
from transformers import Trainer, set_seed
from transformers.training_args import TrainingArguments
import itertools as it
import logging
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from matplotlib import colors

import torch
from tqdm.notebook import trange
import scanpy as sc 
import anndata as ad
# import scib

from geneformer import DataCollatorForCellClassification
from geneformer.tokenizer import TOKEN_DICTIONARY_FILE
from geneformer.in_silico_perturber import pad_tensor_list, quant_layers

sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=200, facecolor='white')

# load token dictionary (Ensembl IDs:token)
with open(TOKEN_DICTIONARY_FILE, "rb") as f:
    gene_token_dict = pickle.load(f)

torch.cuda.memory_allocated()

#### Function for loading pretrained or fine-tuned model

def load_model(model_directory, model_type, num_classes=None):
        if model_type == "Pretrained":
            model = BertForMaskedLM.from_pretrained(model_directory, 
                                                    output_hidden_states=True, 
                                                    output_attentions=False)
        elif model_type == "GeneClassifier":
            model = BertForTokenClassification.from_pretrained(model_directory,
                                                    num_labels=num_classes,
                                                    output_hidden_states=True, 
                                                    output_attentions=False)
        elif model_type == "CellClassifier":
            model = BertForSequenceClassification.from_pretrained(model_directory, 
                                                    num_labels=num_classes,
                                                    output_hidden_states=True, 
                                                    output_attentions=False)
        else: raise(ValueError)
        # put the model in eval mode for fwd pass
        model.eval()
        model = model.to("cuda:0")
        return model

#### Function for obtaining cell or gene embeddings from pretrained or fine-tuned model

def get_cell_embs_from_model(model_dir,
                            model_type="Pretrained",
                            n_classes=None,
                            input_data=None,
                            filter_feature=None, # Name of feature to use for input data filtering, eg. "celltype", "batch"
                            feature_list=None, # List of categories of the given feature above to retain, eg. ["T", "NK", "B"]
                            layer_from_top=0, # Layer to extract embeddings from; 0 for top layer, -1 for second-to-top, etc.
                            token_dictionary=gene_token_dict,
                            forward_batch_size=12,
                            n_proc=8):
    # Load model
    model = load_model(model_directory=model_dir, model_type=model_type, num_classes=n_classes)
    # Define layer to extact embeddings from
    layer_to_quant = quant_layers(model) + layer_from_top
    
    # If designated, filter dataset
    if filter_feature!=None:
        def filter_states(example):
            return example[filter_feature] in feature_list
        input_data = input_data.filter(filter_states, num_proc=n_proc)
    
    # Calculate minibatch metrics
    total_batch_length = len(input_data)
    if ((total_batch_length-1)/forward_batch_size).is_integer():
        forward_batch_size = forward_batch_size-1
    max_len = max(input_data["length"])
    
    # Extract embeddings
    cell_embs_list = []
    for i in range(0, total_batch_length, forward_batch_size):
        max_range = min(i+forward_batch_size, total_batch_length)
                
        state_minibatch = input_data.select([i for i in range(i, max_range)])
        state_minibatch.set_format(type="torch")
            
        input_data_minibatch = state_minibatch["input_ids"]
        input_data_minibatch = pad_tensor_list(input_data_minibatch, max_len, token_dictionary)

        with torch.no_grad():
            outputs = model(input_ids = input_data_minibatch.to("cuda"))
            
        embs_i = outputs.hidden_states[layer_to_quant]
        cell_embs_i = torch.mean(embs_i,dim=[1]).cpu().detach().numpy()
        cell_embs_list += [cell_embs_i]
        
        del outputs
        del state_minibatch
        del input_data_minibatch
        del embs_i
        torch.cuda.empty_cache()
     
    cell_embs = np.concatenate(cell_embs_list)    
    return cell_embs

#### Function for computing metrics on inference dataset

def compute_inference_metrics(pred):
    preds = pred.predictions.argmax(-1)
    return {'pred_label': preds}

#### Function for randomly splitting a single dataset into training set and evaluation set

def preprocess_split_dataset(train_dataset, 
                             eval_size=0.33, # proportion of dataset to use as evaluation set
                             n_proc=8,
                             filter_cell_types=False
                             ):
#     celltype_counter = Counter(train_dataset["cell_type"])
#     total_cells = sum(celltype_counter.values())
#     cells_to_keep = [k for k,v in celltype_counter.items() if v>(0.005*total_cells)]
#     def if_not_rare_celltype(example):
#         return example["cell_type"] in cells_to_keep
#     train_subset = train_dataset.filter(if_not_rare_celltype, num_proc=n_proc) # dataset after filtering low proportion cell types
      
    # shuffle datasets and rename columns
    trainset_shuffled = train_dataset.shuffle(seed=42)
    trainset_shuffled = trainset_shuffled.rename_column("cell_type","label")

    # create dictionary of cell types : label ids
    target_names = list(Counter(trainset_shuffled["label"]).keys()) # List of cell types
    target_name_id_dict = dict(zip(target_names,[i for i in range(len(target_names))]))
    
    # change labels to numerical ids
    def classes_to_ids(example):
        example["label"] = target_name_id_dict[example["label"]]
        return example
    labeled_trainset = trainset_shuffled.map(classes_to_ids, num_proc=n_proc)
    
    # create train/eval splits
    train_size = 1 - eval_size
    labeled_train_split = labeled_trainset.select([i for i in range(0,round(len(labeled_trainset)*train_size))])
    labeled_eval_split = labeled_trainset.select([i for i in range(round(len(labeled_trainset)*train_size),len(labeled_trainset))])
    
    # filter dataset for cell types in corresponding training set
    # in case some cell types are only found in the evaluation set but not in the training set
    if filter_cell_types:
        trained_labels = list(Counter(labeled_train_split["label"]).keys())
        def if_trained_label(example):
            return example["label"] in trained_labels
        labeled_eval_split = labeled_eval_split.filter(if_trained_label, num_proc=n_proc)

    return labeled_train_split, labeled_eval_split, target_name_id_dict

#### Function for preprocessing a given training set and evaluation set (two individual datasets)

def preprocess_cross_dataset(train_set, eval_set, n_proc=8):
    # shuffle datasets and rename columns
    train_set = train_set.shuffle(seed=42)
    eval_set = eval_set.shuffle(seed=42)
    train_set = train_set.rename_column("cell_type","label")
    eval_set = eval_set.rename_column("cell_type","label")
    
    # create dictionary of cell types : label ids
    target_names = list(set(list(Counter(train_set["label"]).keys()) + list(Counter(eval_set["label"]).keys())))
    target_name_id_dict = dict(zip(target_names,[i for i in range(len(target_names))]))
    
    # change labels to numerical ids
    def classes_to_ids(example):
        example["label"] = target_name_id_dict[example["label"]]
        return example
    labeled_train_set = train_set.map(classes_to_ids, num_proc=n_proc)
    labeled_eval_set = eval_set.map(classes_to_ids, num_proc=n_proc)
    
    return labeled_train_set, labeled_eval_set, target_name_id_dict

#### Function for computing metrics when training

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # calculate accuracy and macro f1 using sklearn's function
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro')
    weighted_f1 = f1_score(labels, preds, average='weighted')
    macro_precision = precision_score(labels, preds, average='macro')
    weighted_precision = precision_score(labels, preds, average='weighted')
    macro_recall = recall_score(labels, preds, average='macro')
    weighted_recall = recall_score(labels, preds, average='weighted')
    return {
      'accuracy': acc,
      'macro_f1': macro_f1,
      'weighted_f1': weighted_f1,
      'macro_precision': macro_precision,
      'weighted_precision': weighted_precision,
      'macro_recall': macro_recall,
      'weighted_recall': weighted_recall
    }

# set model parameters
# max input size
max_input_size = 2 ** 11  # 2048

# set training hyperparameters
# max learning rate
max_lr = 5e-5
# how many pretrained layers to freeze
freeze_layers = 0
# number gpus
num_gpus = 1
# number cpu cores
num_proc = 8

# learning schedule
lr_schedule_fn = "linear"
# warmup steps
warmup_steps = 500
# optimizer
optimizer = "adamw_torch"
# optimizer = "sgd"
# optimizer = "lion"

#### Function for fine-tune training

def train_cell_classifier(train_dataset, 
                          eval_dataset,
                          target_name_id_dict,
                          run_name, 
                          geneformer_batch_size=12, # batch size for training and eval
                          epochs=10,                # number of epochs
                          seed=42                   # random seed to use for training
                         ):
    # set logging steps
    logging_steps = round(len(train_dataset)/geneformer_batch_size/10) # Number of mini-batches/10
    
    # reload pretrained model
    set_seed(seed=seed)
    model = BertForSequenceClassification.from_pretrained("/gpfs/gibbs/pi/zhao/kl945/Geneformer/pretrained_model/", # "/path/to/pretrained_model/"
                                                      num_labels=len(target_name_id_dict.keys()),
                                                      output_attentions = False,
                                                      output_hidden_states = False).to("cuda")
    
    # define output directory path
    current_date = datetime.datetime.now()
    datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
    output_dir = f"/gpfs/gibbs/pi/zhao/tl688/cpsc_finalproject/genept_data/geneformer_code/finetuned_models/{datestamp}_geneformer_CellClassifier_{run_name}_S{seed}_L{max_input_size}_B{geneformer_batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_E{epochs}_O{optimizer}_F{freeze_layers}/"
    
    # ensure not overwriting previously saved model
    saved_model_test = os.path.join(output_dir, f"pytorch_model.bin")
    if os.path.isfile(saved_model_test) == True:
        raise Exception("Model already saved to this directory.")

    # make output directory
    subprocess.call(f'mkdir {output_dir}', shell=True)
    
    # set training arguments
    training_args = {
        "learning_rate": max_lr,
        "do_train": True,
        "do_eval": True,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_steps": logging_steps,
        "group_by_length": True,
        "length_column_name": "length",
        "disable_tqdm": False,
        "lr_scheduler_type": lr_schedule_fn,
        "warmup_steps": warmup_steps,
        "weight_decay": 0.001,
        "per_device_train_batch_size": geneformer_batch_size,
        "per_device_eval_batch_size": geneformer_batch_size,
        "num_train_epochs": epochs,
        "load_best_model_at_end": True,
        "output_dir": output_dir,
        "optim":optimizer
    }
    
    training_args_init = TrainingArguments(**training_args)
    

    # create the trainer
    trainer = Trainer(
        model=model,
        args=training_args_init,
        data_collator=DataCollatorForCellClassification(),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    # train the cell type classifier
    trainer.train()
    training_mem = torch.cuda.memory_allocated()
    print(f'Total memory allocated after training: {training_mem}')
    predictions = trainer.predict(eval_dataset)
    predictions.metrics['training_memory_allocated'] = training_mem
    
    with open(f"{output_dir}predictions.p", "wb") as fp:
        pickle.dump(predictions, fp)
    trainer.save_metrics("eval",predictions.metrics)
    trainer.save_model(output_dir)
    
    return datestamp

class ClearCache:
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()


def train_cell_classifier_no_cache(train_dataset, 
                          eval_dataset, 
                          target_name_id_dict,
                          run_name, 
                          geneformer_batch_size=12, # batch size for training and eval
                          epochs=10,                # number of epochs
                          seed=42                   # random seed to use for training
                         ):
    with ClearCache():
        # set logging steps
        logging_steps = round(len(train_dataset)/geneformer_batch_size/10) # Number of mini-batches/10
    
        # reload pretrained model
        set_seed(seed=seed)
        model = BertForSequenceClassification.from_pretrained("../pretrained_model/", # "/path/to/pretrained_model/"
                                                      num_labels=len(target_name_id_dict.keys()),
                                                      output_attentions = False,
                                                      output_hidden_states = False).to("cuda")
    
        # define output directory path
        current_date = datetime.datetime.now()
        datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"
        output_dir = f"/gpfs/gibbs/pi/zhao/kl945/Geneformer/finetuned_models/{datestamp}_geneformer_CellClassifier_{run_name}_S{seed}_L{max_input_size}_B{geneformer_batch_size}_LR{max_lr}_LS{lr_schedule_fn}_WU{warmup_steps}_E{epochs}_O{optimizer}_F{freeze_layers}/"
    
        # ensure not overwriting previously saved model
        saved_model_test = os.path.join(output_dir, f"pytorch_model.bin")
        if os.path.isfile(saved_model_test) == True:
            raise Exception("Model already saved to this directory.")

        # make output directory
        subprocess.call(f'mkdir {output_dir}', shell=True)
    
        # set training arguments
        training_args = {
        "learning_rate": max_lr,
        "do_train": True,
        "do_eval": True,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_steps": logging_steps,
        "group_by_length": True,
        "length_column_name": "length",
        "disable_tqdm": False,
        "lr_scheduler_type": lr_schedule_fn,
        "warmup_steps": warmup_steps,
        "weight_decay": 0.001,
        "per_device_train_batch_size": geneformer_batch_size,
        "per_device_eval_batch_size": geneformer_batch_size,
        "num_train_epochs": epochs,
        "load_best_model_at_end": True,
        "output_dir": output_dir,
        }
    
        training_args_init = TrainingArguments(**training_args)

        # create the trainer
        trainer = Trainer(
        model=model,
        args=training_args_init,
        data_collator=DataCollatorForCellClassification(),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
        )
        # train the cell type classifier
        trainer.train()
        training_mem = torch.cuda.memory_allocated()
        print(f'Total memory allocated after training: {training_mem}')
        predictions = trainer.predict(eval_dataset)
        predictions.metrics['training_memory_allocated'] = training_mem
    
        with open(f"{output_dir}predictions.p", "wb") as fp:
            pickle.dump(predictions, fp)
        trainer.save_metrics("eval",predictions.metrics)
        trainer.save_model(output_dir)
    
    return datestamp

# zebrafish

dataset = load_from_disk('./data/new/zebrafish.dataset/')
run_name = 'zebrafish'
print(dataset)
Counter(dataset['cell_type'])

# Split into train and eval sets
train_dataset, eval_dataset, target_name_id_dict = preprocess_split_dataset(dataset, eval_size=0.33)
with open(f'../data/datasets/splits/{run_name}_trainset.p', 'wb') as f:
    pickle.dump(train_dataset, f)
with open(f'../data/datasets/splits/{run_name}_evalset.p', 'wb') as f:
    pickle.dump(eval_dataset, f)
    
label_dict = {v:k for k, v in target_name_id_dict.items()}
# Train fine-tuning model
print(f'Initial memory allocated: {torch.cuda.memory_allocated()}')
start_time = time.time()

train_cell_classifier(train_dataset=train_dataset, eval_dataset=eval_dataset, run_name=run_name, geneformer_batch_size=12,target_name_id_dict=target_name_id_dict)
training_memory = torch.cuda.memory_allocated()
print(f'Total memory allocated after evaluation: {training_memory}')

end_time = time.time()
run_time = round(end_time - start_time, 3)
print(f'Total running time: {run_time}s')

# Inspect classification results
model_path = "./finetuned_models/240809_geneformer_CellClassifier_zebrafish_S42_L2048_B12_LR5e-05_LSlinear_WU500_E10_Oadamw_torch_F0/"

pred_res = pickle.load(open(f'{model_path}/predictions.p', 'rb'))

# Assign predicted cell type labels
pred_idx = pred_res.predictions.argmax(-1)
pred_labels = [label_dict[i] for i in pred_idx]

# Collect evaluation results and training info
train_eval_info = pred_res.metrics
train_eval_info['run_time'] = run_time
train_eval_info['memory_allocated'] = training_memory
with open('./finetuned_models/train_eval_info.p', 'wb') as f:
    pickle.dump(train_eval_info, f)
    