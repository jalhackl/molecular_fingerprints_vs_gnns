import rdkit

import pandas as pd
import numpy as np
from copy import deepcopy

from torch_geometric.datasets import MoleculeNet
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import AllChem

import torch_geometric.utils


import torch
from torch.nn import Linear
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool

from torch_geometric.data import DataLoader



def create_fingerprint_sets(datasets, radius=2, fpSize = 1024, create_count_fp = True):
    new_datasets = []
    for dataset in datasets:
        new_dataset = []
        fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize = fpSize)

        curr_dataset = [dataset.dataset[i] for i in dataset.indices]

        smilestr = [[x.smiles, x.y] for x in curr_dataset]
        for entry in smilestr:

            if create_count_fp:
                
                new_fp = fpgen.GetCountFingerprint(Chem.MolFromSmiles(entry[0])).ToList()

            else:
                new_fp = fpgen.GetFingerprint(Chem.MolFromSmiles(entry[0])).ToList()

            new_dataset.append([torch.FloatTensor(new_fp), entry[1]])

        new_datasets.append(new_dataset)

    return new_datasets

    


def create_one_morgan_fingerprint_from_dataset(data, radius=2, fpSize = 1024, create_count_fp = True):
    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize = fpSize)

    smiles_list = []
    for index, entry in  data.iterrows():
        smiles_list.append([entry[0], Chem.MolFromSmiles(entry["smiles"])])


        

    if create_count_fp:
        morgan_count_list = []
        for index, entry in enumerate(smiles_list):
            morgan_count_list.append([entry[0], 
                                ''.join([str(x) for x in fpgen.GetCountFingerprint(entry["smiles"]).ToList()])
                                ])
            
        return smiles_list, morgan_count_list


    
    else:

        morgan_list = []
        for index, entry in enumerate(smiles_list):
            morgan_list.append([entry[0], 
                                fpgen.GetFingerprint(entry["smiles"]).ToBitString() ])
        
        return smiles_list, morgan_list




def create_one_morgan_fingerprint(data, radius=2, fpSize = 1024, create_count_fp = True):
    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize = fpSize)

    smiles_list = []
    for index, entry in  data.iterrows():
        smiles_list.append([entry[0], Chem.MolFromSmiles(entry["smiles"])])


        

    if create_count_fp:
        morgan_count_list = []
        for index, entry in enumerate(smiles_list):
            morgan_count_list.append([entry[0], 
                                ''.join([str(x) for x in fpgen.GetCountFingerprint(entry["smiles"]).ToList()])
                                ])
            
        return smiles_list, morgan_count_list


    
    else:

        morgan_list = []
        for index, entry in enumerate(smiles_list):
            morgan_list.append([entry[0], 
                                fpgen.GetFingerprint(entry["smiles"]).ToBitString() ])
        
        return smiles_list, morgan_list





def create_morgan_fingerprint(data, radius=2, fpSize = 1024, create_count_fp = True):
    fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize = fpSize)

    smiles_list = []
    for index, entry in  data.iterrows():
        smiles_list.append([entry[0], Chem.MolFromSmiles(entry["smiles"])])

    morgan_list = []
    for index, entry in enumerate(smiles_list):
        morgan_list.append([entry[0], 
                            fpgen.GetFingerprint(entry["smiles"]).ToBitString() ])
        

    if create_count_fp:
        morgan_count_list = []
        for index, entry in enumerate(smiles_list):
            morgan_count_list.append([entry[0], 
                                ''.join([str(x) for x in fpgen.GetCountFingerprint(entry["smiles"]).ToList()])
                                ])
            
        return smiles_list, morgan_list, morgan_count_list
    
    else:
        
        return smiles_list, morgan_list



def create_pytorch_graph(data):
    pytorch_graph_list = []
    for index, entry in  data.iterrows():
        new_graph = torch_geometric.utils.from_smiles(entry["smiles"])
        pytorch_graph_list.append([entry[0], 
                            new_graph])
    return pytorch_graph_list
        

def process_pytorch_graph(pytorch_graph_list, data_y):
    data = [x[1] for x in pytorch_graph_list]
    for ie, entry in enumerate(data):
        entry.y = torch.tensor([[float(data_y[ie])]])
    return data

        

def create_train_test_graphs(data, train_percentage = 0.8, test_percentage = None, apply_scaffold_split = False):
    if test_percentage is None:
        test_percentage = 1 - train_percentage

    val_percentage = 1 - train_percentage - test_percentage
    if apply_scaffold_split:
        import dgl
        from dgllife.utils import ScaffoldSplitter

        class ScaffoldList(list):
            def __init__(self, moldata):
                super().__init__(moldata)
                self.smiles = [x.smiles for x in moldata]

        data = ScaffoldList(data)
        train_dataset, test_dataset, val_dataset = ScaffoldSplitter.train_val_test_split(data, mols=None, sanitize=True, frac_train=train_percentage, frac_val=1-train_percentage-val_percentage, frac_test=val_percentage, log_every_n=1000, scaffold_func='decompose')

    else:
        data_size = len(data)
        if train_percentage + test_percentage == 1:
            
            train_size = int(train_percentage * data_size)
            test_size = data_size - train_size
            val_size = 0

        else:
            train_size = int(train_percentage * data_size)
            test_size = int(test_percentage * data_size) 
            val_size = data_size - train_size - test_size 

        train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, test_size, val_size])

    return train_dataset, test_dataset, val_dataset 


def create_train_test_graphs_plus_fingerprints(data, train_percentage = 0.8, test_percentage = None, apply_scaffold_split = False):
    if test_percentage is None:
        test_percentage = 1 - train_percentage

    val_percentage = 1 - train_percentage - test_percentage
    if apply_scaffold_split:
        import dgl
        from dgllife.utils import ScaffoldSplitter

        class ScaffoldList(list):
            def __init__(self, moldata):
                super().__init__(moldata)
                self.smiles = [x.smiles for x in moldata]

        data = ScaffoldList(data)
        train_dataset, test_dataset, val_dataset = ScaffoldSplitter.train_val_test_split(data, mols=None, sanitize=True, frac_train=train_percentage, frac_val=1-train_percentage-val_percentage, frac_test=val_percentage, log_every_n=1000, scaffold_func='decompose')

    else:
        data_size = len(data)
        if train_percentage + test_percentage == 1:
            
            train_size = int(train_percentage * data_size)
            test_size = data_size - train_size
            val_size = 0

        else:
            train_size = int(train_percentage * data_size)
            test_size = int(test_percentage * data_size) 
            val_size = data_size - train_size - test_size 

        train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, test_size, val_size])

    return train_dataset, test_dataset, val_dataset 


def create_dataloader(train_dataset, test_dataset, batch_size=64):
    loader = DataLoader(train_dataset, 
                    batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, 
                            batch_size=batch_size, shuffle=True)
    return loader, test_loader


def create_dataloader_val(train_dataset, test_dataset, val_dataset, batch_size=64):
    loader = DataLoader(train_dataset, 
                    batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, 
                            batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size, shuffle=True)
    return loader, test_loader, val_loader



def plot_losses(losses, val_losses, log_time):
    import seaborn as sns
    import pandas as pd

    losses_float = [float(loss) for loss in losses] 
    loss_indices = [i*log_time for i, l in enumerate(losses_float)] 

    
    val_losses_float = [float(loss) for loss in val_losses] 
    val_loss_indices = [i*log_time for i, l in enumerate(losses_float)] 

    data_loss = pd.DataFrame({'epoch': loss_indices, 'loss': losses_float, 'index2': val_loss_indices, 'val_loss': val_losses_float})

    data_loss.set_index("epoch",inplace=True)
    plt = sns.lineplot(data=data_loss[["loss", "val_loss"]])
            
    