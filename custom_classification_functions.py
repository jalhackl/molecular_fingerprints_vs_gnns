import pandas as pd
import numpy as np
from copy import deepcopy

from torch_geometric.datasets import MoleculeNet


def load_hiv():
    processed_data = MoleculeNet(root=".", name="HIV") 
    data = pd.read_csv("./hiv/raw/HIV.csv")
    data_y = data["HIV_active"]
    return data, data_y



