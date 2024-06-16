from dgllife.utils import smiles_to_complete_graph
from dgllife.utils import CanonicalAtomFeaturizer
from dgllife.utils import CanonicalBondFeaturizer



def create_dgl_graph(data, data_y):
    atom_featurizer = CanonicalAtomFeaturizer()

    dgl_graph_list = []
    for index, entry in  data.iterrows():
        try:
            new_graph = smiles_to_complete_graph(entry["smiles"], node_featurizer=atom_featurizer)
            dgl_graph_list.append([entry[0], 
                                new_graph, float(data_y[index])])
        except:
            print("There was a problem with ")
            print(entry[-1])
            print(f" at index {index}")

    return dgl_graph_list



def process_dgl_graph(dgl_graph_list):
    data = [x[1:] for x in dgl_graph_list]
    return data



def create_dgl_dataloader(train_dataset, test_dataset, batch_size=64):
    from dgl.dataloading import GraphDataLoader

    loader = GraphDataLoader(train_dataset, 
                    batch_size=batch_size, shuffle=True)
    test_loader = GraphDataLoader(test_dataset, 
                            batch_size=batch_size, shuffle=True)
    return loader, test_loader

def create_dgl_dataloader_val(train_dataset, test_dataset, val_dataset, batch_size=64):
    from dgl.dataloading import GraphDataLoader

    loader = GraphDataLoader(train_dataset, 
                    batch_size=batch_size, shuffle=True)
    test_loader = GraphDataLoader(test_dataset, 
                            batch_size=batch_size, shuffle=True)
    val_loader = GraphDataLoader(val_dataset, 
                            batch_size=batch_size, shuffle=True)
    return loader, test_loader, val_loader




def create_dgl_sets(datasets):
    new_datasets = []
    for dataset in datasets:
        new_dataset = []
        #fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize = fpSize)

        curr_dataset = [dataset.dataset[i] for i in dataset.indices]
        
        import pandas as pd
        #curr_dataset = pd.DataFrame(curr_dataset)

        curr_smiles = pd.DataFrame([x.smiles for x in curr_dataset], columns=["smiles"])

        #smilestr = [[x.smiles, x.y] for x in curr_dataset]
        #smilestr = [x.smiles for x in curr_dataset]
        data_y = [x.y for x in curr_dataset]


        #for entry in smilestr:
        dgl_graph_list = create_dgl_graph(curr_smiles, data_y)
        new_dataset = process_dgl_graph(dgl_graph_list)

            #new_dataset.append([torch.FloatTensor(new_fp), entry[1]])

        new_datasets.append(new_dataset)

    return new_datasets

