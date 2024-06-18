from memory_profiler import memory_usage
save = True

import time
import sys
import os
from os import sys, path

print(len(sys.argv))
assert len(sys.argv) == 4 or len(sys.argv) == 6 or len(sys.argv) == 7 or len(sys.argv) == 10, "parameters needed: rna path, atac path, result folder"
rna_path = str(sys.argv[1])
atac_path = str(sys.argv[2])
result_folder = str(sys.argv[3])
int_data_path = path.join(str(sys.argv[3]), 'tmp')

stage1_lr = 0.01
if result_folder.split('/')[-2] in ['MOp']:
    stage1_lr = 0.001
stage3_lr = 0.01
if result_folder.split('/')[-2] in ['MOp', 'HSPC_paired', 'MouseEmbryo_paired']:
    stage3_lr = 0.001
if result_folder.split('/')[-3] == 'results_time_mem_MOp':
    stage1_lr = 0.001
    stage3_lr = 0.001
nepoch = 20 #default values
if len(sys.argv) == 10:
    stage1_lr = float(sys.argv[7])
    stage3_lr = float(sys.argv[8])
    nepoch = int(sys.argv[9])
print(stage1_lr, stage3_lr, nepoch)
import scanpy
from scipy.sparse import csc_matrix, csr_matrix, save_npz
from scipy.io import mmwrite, mmread
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from datetime import datetime
sys.path.append('your_path_of_scJoint')
from util.trainingprocess_stage1 import TrainingProcessStage1
from util.trainingprocess_stage3 import TrainingProcessStage3
from util.knn import KNN

def read_txt_np(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
        return np.array(lines)

#data_path = "../data/BMMC_processed_s3d7"
#result_folder = "../results/scJoint_BMMC_s3d7"
# int_data_path = '/gpfs/gibbs/pi/zhao/xs272/Multiomics/scJoint/my_data'

def run_scJoint(rna_path, atac_path, result_folder, subset_rna, subset_atac, rna_new_annot, stage1_lr, stage3_lr, nepoch):
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    if not os.path.exists(int_data_path):
        os.makedirs(int_data_path)
    if rna_path.endswith('mtx'):
        rna_counts = csr_matrix(mmread(rna_path))
        rna_path = '/'.join(rna_path.split('/')[:-1])
    else:
        rna_counts = csr_matrix(mmread(path.join(rna_path, 'counts.mtx')))
    if atac_path.endswith('mtx'):
        atac_counts = csr_matrix(mmread(atac_path))
        atac_path = '/'.join(atac_path.split('/')[:-1])
    else:
        atac_counts = csr_matrix(mmread(path.join(atac_path, 'counts.mtx')))
    # rna_gene_names = np.loadtxt(path.join(rna_path, 'genes.txt'), dtype=object, delimiter='DONTWANTSPACEASDILIMITERS')
    # atac_gene_names = np.loadtxt(path.join(atac_path, 'genes.txt'), dtype=object, delimiter='SPACEISPARTOFTHENAME')
    # rna_cell_names = np.loadtxt(path.join(rna_path, 'cells.txt'), dtype=object, delimiter='DONTWANTSPACEASDILIMITERS')
    # atac_cell_names = np.loadtxt(path.join(atac_path, 'cells.txt'), dtype=object, delimiter='SPACEISPARTOFTHENAME')
    # rna_label = np.loadtxt(path.join(rna_path, 'annotations.txt'), dtype=object, delimiter='SPACEISPARTOFTHENAME')
    # atac_label = np.loadtxt(path.join(atac_path, 'annotations.txt'), dtype=object, delimiter='SPACESHOULDB')
    rna_gene_names = read_txt_np(path.join(rna_path, 'genes.txt'))
    atac_gene_names = read_txt_np(path.join(atac_path, 'genes.txt'))
    rna_cell_names = read_txt_np(path.join(rna_path, 'cells.txt'))
    atac_cell_names = read_txt_np(path.join(atac_path, 'cells.txt'))
    if rna_new_annot is not None:
        rna_label = read_txt_np(rna_new_annot)
    else:
        rna_label = read_txt_np(path.join(rna_path, 'annotations.txt'))
    atac_label = read_txt_np(path.join(atac_path, 'annotations.txt'))
    # rna_meta = np.load(path.join(data_path, 'RNA', 'metadata.npz'), allow_pickle=True)
    # atac_meta = np.load(path.join(data_path, 'ATAC', 'metadata.npz'), allow_pickle=True)
    # get names
    # rna_gene_names = rna_meta['features']
    # atac_gene_names = np.load(path.join(data_path, 'ATAC', 'activity.features.npy'), allow_pickle=True)

    ## subset
    if subset_rna is not None:
        subset_rna_barcodes = read_txt_np(subset_rna)
        # cell_names, idx_bc_rna, idx_bc = np.intersect1d(rna_cell_names, subset_rna_barcodes, return_indices=True)
        ids_series = pd.Series(np.arange(len(rna_cell_names)), index=rna_cell_names)
        idx_bc_rna = ids_series[subset_rna_barcodes]
        rna_counts = rna_counts[idx_bc_rna, :]
        rna_label = rna_label[idx_bc_rna]
        rna_cell_names = subset_rna_barcodes

    if subset_atac is not None:
        subset_atac_barcodes = read_txt_np(subset_atac)
        # cell_names, idx_bc_atac, idx_bc = np.intersect1d(atac_cell_names, subset_atac_barcodes, return_indices=True)
        ids_series = pd.Series(np.arange(len(atac_cell_names)), index=atac_cell_names)
        idx_bc_atac = ids_series[subset_atac_barcodes]
        atac_counts = atac_counts[idx_bc_atac, :]
        atac_label = atac_label[idx_bc_atac]  
        atac_cell_names = subset_atac_barcodes

    rna_cell_filter = np.array((rna_counts > 0).sum(1)).flatten() >= 200
    rna_counts = rna_counts[rna_cell_filter, :]
    rna_label = rna_label[rna_cell_filter]
    rna_cell_names = rna_cell_names[rna_cell_filter]
    rna_gene_filter = np.array((rna_counts > 0).sum(0)).flatten() >= 3
    rna_counts = rna_counts[:, rna_gene_filter]
    rna_gene_names = rna_gene_names[rna_gene_filter]

    atac_cell_filter = np.array((atac_counts > 0).sum(1)).flatten() >= 200
    atac_counts = atac_counts[atac_cell_filter, :]
    atac_label = atac_label[atac_cell_filter]
    atac_cell_names = atac_cell_names[atac_cell_filter]
    atac_gene_filter = np.array((atac_counts > 0).sum(0)).flatten() >= 3
    atac_counts = atac_counts[:, atac_gene_filter]
    atac_gene_names = atac_gene_names[atac_gene_filter]

    gene_names, idx_rna, idx_atac = np.intersect1d(rna_gene_names, atac_gene_names, return_indices=True)
    rna_counts = rna_counts[:, idx_rna]
    atac_counts = atac_counts[:, idx_atac]

    # import pdb;pdb.set_trace()

#     assert not os.path.exists(path.join(int_data_path, 'rna_data.npz')), "data file exists!"
#     assert not os.path.exists(path.join(int_data_path, 'atac_data.npz')), "data file exists!"
    Path(int_data_path).mkdir(parents=True, exist_ok=True)    
    # save to npz
    save_npz(path.join(int_data_path, 'rna_data.npz'), rna_counts)
    save_npz(path.join(int_data_path, 'atac_data.npz'), atac_counts)
    # save atac cell names
    np.savetxt(path.join(int_data_path, 'atac_cells.txt'), atac_cell_names, fmt='%s')
    # write label files
    # cell_label = rna_meta['annotation']
    cell_label = rna_label
    label_idx_mapping = {}
    unique_labels = np.unique(cell_label)
    for i, name in enumerate(unique_labels):
        label_idx_mapping[name] = i
    print(label_idx_mapping)
    with open(path.join(int_data_path, "label_to_idx.txt"), "w") as fp:
        for key in sorted(label_idx_mapping):
            fp.write(key + "\t" + str(label_idx_mapping[key]) + '\n')
    # write label files
    with open(path.join(int_data_path, 'rna_label.txt'), 'w') as rna_label_f:
        for label in cell_label:
            rna_label_f.write(str(label_idx_mapping[label]) + '\n')
    with open(path.join(int_data_path, 'atac_label.txt'), 'w') as atac_label_f:
        for label in atac_label:
            if label in label_idx_mapping.keys():
                atac_label_f.write(str(label_idx_mapping[label]) + '\n')
            else:
                atac_label_f.write('-1\n')
    if result_folder[-1] != '/':
        result_folder += '/'
    # np.savetxt(result_folder+'/rna.dim', rna_counts.shape)
    main_scJoint(
        number_of_class=len(unique_labels),
        input_size=len(gene_names),
        rna_paths=[path.join(int_data_path, 'rna_data.npz')],
        rna_labels=[path.join(int_data_path, 'rna_label.txt')],
        atac_paths=[path.join(int_data_path, 'atac_data.npz')],
        atac_labels=[path.join(int_data_path, 'atac_label.txt')],
        result_folder=result_folder,
        stage1_lr=stage1_lr, 
        stage3_lr=stage3_lr, 
        nepoch=nepoch
    )

class Config(object):
    def __init__(self, number_of_class, input_size, rna_paths, rna_labels, atac_paths, atac_labels, stage1_lr, stage3_lr, nepoch):
        self.use_cuda = True
        self.threads = 1

        if not self.use_cuda:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:0')

        # DB info
        self.number_of_class = number_of_class
        self.input_size = input_size
        self.rna_paths = rna_paths
        self.rna_labels = rna_labels
        self.atac_paths = atac_paths
        self.atac_labels = atac_labels #Optional. If atac_labels are provided, accuracy after knn would be provided.
        self.rna_protein_paths = []
        self.atac_protein_paths = []

        # Training config
        self.batch_size = 256
        self.lr_stage1 = stage1_lr
        self.lr_stage3 = stage3_lr
        # self.lr_decay_epoch = 20
        # self.epochs_stage1 = 20
        # self.epochs_stage3 = 20
        self.lr_decay_epoch = nepoch
        self.epochs_stage1 = nepoch
        self.epochs_stage3 = nepoch
        self.p = 0.8
        self.embedding_size = 64
        self.momentum = 0.9
        self.center_weight = 1
        self.with_crossentorpy = True
        self.seed = 1
        self.checkpoint = ''

def main_scJoint(number_of_class, input_size, rna_paths, rna_labels, atac_paths, atac_labels, result_folder, stage1_lr, stage3_lr, nepoch):
    # hardware constraint for speed test
    torch.set_num_threads(1)

    os.environ['OMP_NUM_THREADS'] = '1'

    # initialization
    config = Config(number_of_class, input_size, rna_paths, rna_labels, atac_paths, atac_labels, stage1_lr, stage3_lr, nepoch)
    torch.manual_seed(config.seed)
    print('Start time: ', datetime.now().strftime('%H:%M:%S'))


    # stage1 training
    print('Training start [Stage1]')
    model_stage1= TrainingProcessStage1(config)
    for epoch in range(config.epochs_stage1):
        print('Epoch:', epoch)
        model_stage1.train(epoch)

    print('Write embeddings')
    model_stage1.write_embeddings(result_folder)
    print('Stage 1 finished: ', datetime.now().strftime('%H:%M:%S'))

    # KNN
    print('KNN')
    KNN(config, neighbors = 30, knn_rna_samples=20000, output_folder=result_folder)
    print('KNN finished: ', datetime.now().strftime('%H:%M:%S'))


    # stage3 training
    print('Training start [Stage3]')
    model_stage3 = TrainingProcessStage3(config, result_folder)
    for epoch in range(config.epochs_stage3):
        print('Epoch:', epoch)
        model_stage3.train(epoch)

    print('Write embeddings [Stage3]')
    model_stage3.write_embeddings(result_folder)
    print('Stage 3 finished: ', datetime.now().strftime('%H:%M:%S'))

    # KNN
    print('KNN stage3')
    KNN(config, neighbors = 30, knn_rna_samples=20000, output_folder=result_folder)
    print('KNN finished: ', datetime.now().strftime('%H:%M:%S'))

def main(): 
    
    start = time.time()
    
    result_folder = str(sys.argv[3])
    if len(sys.argv) >= 6:
        subset_rna = str(sys.argv[4])
        if subset_rna == '!':
            subset_rna = None
        subset_atac = str(sys.argv[5])
        if subset_atac == '!':
            subset_atac = None
    else:
        subset_rna = None
        subset_atac = None
    if len(sys.argv) >= 7:
        rna_new_annot = sys.argv[6]
        if rna_new_annot == '!':
            rna_new_annot = None
    else:
        rna_new_annot = None
    
    run_scJoint(rna_path, atac_path, result_folder, subset_rna, subset_atac, rna_new_annot, stage1_lr, stage3_lr, nepoch)
    # save results
    if result_folder[-1] != '/':
        result_folder += '/'

    if save:
        # prob matrix
        prob = pd.read_table(result_folder+'atac_data_knn_probs_all.txt', sep='\s', header=None)
    #     if subset_atac is not None:
    #         cells = read_txt_np(subset_atac)
    #     else:
    #         cells = read_txt_np(path.join(atac_path, 'cells.txt'))
        cells = read_txt_np('%s/atac_cells.txt' % int_data_path)
        prob.index = cells
        ct_dic = pd.read_table(path.join(int_data_path, "label_to_idx.txt"), sep='\t', header=None).sort_values(1)
        prob.columns = ct_dic[0]
        prob.to_csv(result_folder+'prob.csv')
        # predicted label
        pred = prob.idxmax(axis=1)
        pred.to_csv(result_folder+'pred.csv')

    end = time.time()
    print('Running time: %.2f sec' % (end-start))
    with open(path.join(result_folder, 'time.txt'), "w") as file:
        file.write(str(end-start))
    
peak_mem_usage = memory_usage(main, max_iterations=1, max_usage=True)
print('Peak memory usage: %.2f MB' % peak_mem_usage)
with open(path.join(result_folder, 'memory.txt'), "w") as file:
    file.write(str(peak_mem_usage))
