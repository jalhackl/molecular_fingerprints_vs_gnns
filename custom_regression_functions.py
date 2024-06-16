import pandas as pd
import numpy as np
from copy import deepcopy

from torch_geometric.datasets import MoleculeNet


def load_esol(use_molecule_net = True):
    if not use_molecule_net:
        data = "./data_duvenaud/2015-05-24-delaney/delaney-processed.csv"
        data = pd.read_csv(data)
    else:
        processed_data = MoleculeNet(root=".", name="ESOL")
        #the processed ESOL data from MoleculeNet is identical to the one used by Duvenaud 2015
        data = pd.read_csv("./esol/raw/delaney-processed.csv")

    data_y = data["measured log solubility in mols per litre"]
    
    return data, data_y












