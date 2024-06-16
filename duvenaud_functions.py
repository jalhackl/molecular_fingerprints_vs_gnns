#these functions are directly copied from Duvenaud 2015

import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import autograd.numpy as np

def smiles_to_fps(data, fp_length, fp_radius):
    return stringlist2intarray(np.array([smile_to_fp(s, fp_length, fp_radius) for s in data]))

def smile_to_fp(s, fp_length, fp_radius):
    m = Chem.MolFromSmiles(s)
    return (AllChem.GetMorganFingerprintAsBitVect(
        m, fp_radius, nBits=fp_length)).ToBitString()

def stringlist2intarray(A):
    '''This function will convert from a list of strings "10010101" into in integer numpy array.'''
    return np.array([list(s) for s in A], dtype=int)



def build_morgan_fingerprint_fun(fp_length=512, fp_radius=4):

    #def fingerprints_from_smiles(weights, smiles):
    def fingerprints_from_smiles(smiles):
        # Morgan fingerprints don't use weights.
        return fingerprints_from_smiles_tuple(tuple(smiles))

    #@memoize # This wrapper function exists because tuples can be hashed, but arrays can't.
    def fingerprints_from_smiles_tuple(smiles_tuple):
        return smiles_to_fps(smiles_tuple, fp_length, fp_radius)

    return fingerprints_from_smiles


'''
from experiment_scripts/launch_experiments.py
morgan_bounds = dict(fp_length      = [16, 1024],
                     fp_depth       = [1, 4],
                     log_init_scale = [-2, -6],
                     log_step_size  = [-8, -4],
                     log_L2_reg     = [-6, 2],
                     h1_size        = [50, 100],
                     conv_width     = [5, 20])

neural_bounds = dict(fp_length      = [16, 128],   # Smaller upper range.
                     fp_depth       = [1, 4],
                     log_init_scale = [-2, -6],
                     log_step_size  = [-8, -4],
                     log_L2_reg     = [-6, 2],
                     h1_size        = [50, 100],
                     conv_width     = [5, 20])

'''