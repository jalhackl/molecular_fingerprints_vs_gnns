import torch
from torch.nn import Linear
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool

from copy import deepcopy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import *

from custom_general_functions import *



def initialize_regression_model(input_dim, hidden_channels=[64], gcn_layers=4, linear_sizes=[], aggregations=[global_mean_pool, global_max_pool], apply_random_aggregations=False, learning_rate=0.001):

    model = GCN_molecule_regression(input_dim = input_dim, hidden_channels=hidden_channels, gcn_layers=gcn_layers, aggregations=aggregations, apply_random_aggregations=apply_random_aggregations)
    print(model)
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    loss_fn = torch.nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, device, optimizer, loss_fn

#currently, this function is identical to regression
def initialize_classification_model(input_dim, hidden_channels=[64], gcn_layers=4, linear_sizes=[], aggregations=[global_mean_pool, global_max_pool], apply_random_aggregations=False, learning_rate=0.001):

    model = GCN_molecule_classification(input_dim = input_dim, hidden_channels=hidden_channels, gcn_layers=gcn_layers, aggregations=aggregations, apply_random_aggregations=apply_random_aggregations)
    print(model)
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, device, optimizer, loss_fn
    

class GCN_molecule_regression(torch.nn.Module):
    #def __init__(self, hidden_channels=[64], gcn_layers=4, input_dim=pytorch_graph_list[0][-1].x.shape[-1], linear_sizes=[], aggregations=[global_mean_pool, global_max_pool], apply_random_aggregations=False ):
    def __init__(self, input_dim, hidden_channels=[64], gcn_layers=4, linear_sizes=[], aggregations=[global_mean_pool, global_max_pool], apply_random_aggregations=False):

        super(GCN_molecule_regression, self).__init__()

        self.apply_random_aggregations = apply_random_aggregations

        self.convs = torch.nn.ModuleList()

        if self.apply_random_aggregations:
            self.random_readout = torch.nn.Parameter(torch.randn((1, hidden_channels[-1]*len(aggregations))))
            self.aggregations = []

        else:

            if len(hidden_channels) == 1:
                self.convs.append(GCNConv(input_dim, hidden_channels[0]))
                for _ in range(gcn_layers - 1):
                    self.convs.append(GCNConv(hidden_channels[0], hidden_channels[0]))

            else:
                self.convs.append(GCNConv(input_dim, hidden_channels[0]))
            
                for i in range(1, len(hidden_channels)):
                    self.convs.append(GCNConv(hidden_channels[i-1], hidden_channels[i]))


            #self.activation1 = torch.nn.Tanh()
            self.activation1 = torch.nn.ReLU()
            self.activation2 = torch.nn.ReLU()

            self.aggregations = aggregations

        self.additional_layers = torch.nn.ModuleList()
        
        if len(linear_sizes) > 0:
            
            first_additional_channel_dim = hidden_channels[-1]*len(self.aggregations)

            for in_dim, out_dim in zip([first_additional_channel_dim] + linear_sizes[:-1], linear_sizes):

                self.additional_layers.append(Linear(in_dim, out_dim))

            self.out = Linear(linear_sizes[-1], 1)

        else:


            self.out = Linear(hidden_channels[0]*len(aggregations), 1)

    def forward(self, x, edge_index, batch_index):

        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.activation1(x)


        xs = []
        for aggregation in self.aggregations:
            xs.append(aggregation(x, batch_index))

        if not self.apply_random_aggregations:
            x = torch.cat(xs, dim=1)
        else:
            batch_size = torch.max(batch_index).item() + 1
            x = self.random_readout.repeat(batch_size, 1)

        out = x

        for layer in self.additional_layers:
            out = layer(out)
            out = self.activation2(out)

        out = self.out(out)

        return out, x
    


class GCN_molecule_classification(torch.nn.Module):
    #def __init__(self, hidden_channels=[embedding_size], num_classes=1, gcn_layers=4, input_dim=pytorch_graph_list[0][-1].x.shape[-1], linear_sizes=[], aggregations=[global_mean_pool, global_max_pool], apply_random_aggregations=False ):
    def __init__(self, input_dim, hidden_channels=[64], num_classes=1, gcn_layers=4, linear_sizes=[], aggregations=[global_mean_pool, global_max_pool], apply_random_aggregations=False):
 
        super(GCN_molecule_classification, self).__init__()

        self.apply_random_aggregations = apply_random_aggregations

        self.convs = torch.nn.ModuleList()

        if self.apply_random_aggregations:
            self.random_readout = torch.nn.Parameter(torch.randn((1, hidden_channels[-1]*len(aggregations))))
            self.aggregations = []

        else:

            if len(hidden_channels) == 1:
                self.convs.append(GCNConv(input_dim, hidden_channels[0]))
                for _ in range(gcn_layers - 1):
                    self.convs.append(GCNConv(hidden_channels[0], hidden_channels[0]))

            else:
                self.convs.append(GCNConv(input_dim, hidden_channels[0]))
            
                for i in range(1, len(hidden_channels)):
                    self.convs.append(GCNConv(hidden_channels[i-1], hidden_channels[i]))


            #self.activation1 = torch.nn.Tanh()
            self.activation1 = torch.nn.ReLU()
            self.activation2 = torch.nn.ReLU()

            self.aggregations = aggregations

        self.additional_layers = torch.nn.ModuleList()
        
        if len(linear_sizes) > 0:
            
            first_additional_channel_dim = hidden_channels[-1]*len(self.aggregations)

            for in_dim, out_dim in zip([first_additional_channel_dim] + linear_sizes[:-1], linear_sizes):

                self.additional_layers.append(Linear(in_dim, out_dim))

            self.out = Linear(linear_sizes[-1], num_classes)

        else:


            self.out = Linear(hidden_channels[0]*len(aggregations), num_classes)



        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index, batch_index):

        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.activation1(x)

        xs = []
        for aggregation in self.aggregations:
            xs.append(aggregation(x, batch_index))

        if not self.apply_random_aggregations:
            x = torch.cat(xs, dim=1)
        else:
            batch_size = torch.max(batch_index).item() + 1
            x = self.random_readout.repeat(batch_size, 1)


        out = x

        for layer in self.additional_layers:
            out = layer(out)
            out = self.activation2(out)

        out = self.out(out)
        #out = out.sigmoid(x)
        return out, x






def regression_train(model, loader, test_loader, device, loss_fn, optimizer, log_time=10, max_epochs=1000, apply_early_stopping = True, early_stopping_patience = 50, finally_plot_losses = False):
    model.train()
    losses = []
    val_losses = []

    best_loss = float('inf')
    best_model_weights = None

    if apply_early_stopping:
        patience = early_stopping_patience

    for epoch in range(max_epochs):
        running_loss = 0
        running_val_loss=0
        
        model.train()


        for batch in loader:

            batch.to(device)  

            optimizer.zero_grad() 

            pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch) 

            loss = loss_fn(pred, batch.y)     
            loss.backward()  

            optimizer.step() 

            running_loss += loss.item()

        model.eval()
        for test_batch in test_loader:
            test_batch.to(device)  

            pred, embedding = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch) 

            val_loss = loss_fn(pred, test_batch.y)     

            running_val_loss += val_loss.item()

        
        #avg_val_loss = np.mean(val_losses)
        if epoch % log_time == 0:
            print(f"Epoch {epoch} | Train Loss {running_loss/len(loader)} | Validation Loss {running_val_loss/len(test_loader)}")
            losses.append(running_loss/len(loader))
            val_losses.append(running_val_loss/len(test_loader))

        if apply_early_stopping:
            if running_val_loss < best_loss:
                best_loss = running_val_loss
                best_model_weights = deepcopy(model.state_dict())
                patience = early_stopping_patience
            else:
                patience = patience - 1
                if patience == 0:
                    break

    if finally_plot_losses:
        plot_losses(losses, val_losses, log_time)

    return model, best_model_weights, losses, val_losses




def classification_train(model, loader, test_loader, device, loss_fn, optimizer, log_time=10, max_epochs=1000, apply_early_stopping = True, early_stopping_patience = 50, finally_plot_losses = False):
    from scipy.special import expit
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix, accuracy_score

    import warnings
    warnings.filterwarnings("ignore")
    
    model.train()
    losses = []
    val_losses = []

    best_loss = float('inf')
    best_model_weights = None

    if apply_early_stopping:
        patience = early_stopping_patience

    for epoch in range(max_epochs):
        
        val_accuracies = []

        running_loss = 0
        running_val_loss=0
        
        model.train()


        for batch in loader:
        #batch.y = batch.y.float()
        #batch.y = batch.y.squeeze()


            batch.to(device)  

            optimizer.zero_grad() 

            pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch) 

            y_pred = np.round(expit(pred.detach().cpu().numpy().flatten()))


            loss = loss_fn(pred, batch.y)     
            loss.backward()  

            optimizer.step() 

            running_loss += loss.item()


        model.eval()

        predicted_labels = []
        true_labels = []
        #correct = 0
        #total = 0
        for test_batch in test_loader:

            
            test_batch.to(device)  

            pred, embedding = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch) 

            y_pred = np.round(expit(pred.detach().cpu().numpy().flatten()))

            val_accuracies.append(accuracy_score(test_batch.y.tolist(), y_pred.flatten()))

            val_loss = loss_fn(pred, test_batch.y)     

            running_val_loss += val_loss.item()
            #total += test_batch.y.size(0)
            #_, predicted = torch.max(pred, 1)
            #correct += (predicted == test_batch.y).sum().item()

            true_labels.extend(test_batch.y.tolist())
            predicted_labels.extend(y_pred)

        accuracy = np.mean(val_accuracies)

        if epoch % log_time == 0:
            print(f"Epoch {epoch} | Train Loss {running_loss/len(loader)} | Validation Loss {running_val_loss/len(test_loader)} | Validation accuracy {accuracy}")
            losses.append(running_loss/len(loader))
            val_losses.append(running_val_loss/len(test_loader))

        if apply_early_stopping:
            if running_val_loss < best_loss:
                best_loss = running_val_loss
                best_model_weights = deepcopy(model.state_dict())
                patience = early_stopping_patience
            else:
                patience = patience - 1
                if patience == 0:
                    break

    if finally_plot_losses:
        plot_losses(losses, val_losses, log_time)

    return model, best_model_weights, losses, val_losses








def predict_regression(model, test_loader, device, best_model_weights=None, plot_final = True):
    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()

    mse_losses = []
    l1_losses = []
    dfs = []

    #load best model
    if best_model_weights is not None:
        print("best weights loaded")
        model.load_state_dict(best_model_weights)

    for test_batch in test_loader:
        with torch.no_grad():
            test_batch.to(device)
            pred, embed = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch) 
            df = pd.DataFrame()
            df["y_real"] = test_batch.y.tolist()
            df["y_pred"] = pred.tolist()

            df["y_real"] = df["y_real"].apply(lambda row: row[0])
            df["y_pred"] = df["y_pred"].apply(lambda row: row[0])
            dfs.append(df)

            mse_losses.append(mse_loss(pred, test_batch.y))
            l1_losses.append(l1_loss(pred, test_batch.y))

    final_df = pd.concat(dfs)

    if plot_final:
        plt = sns.scatterplot(data=final_df, x="y_real", y="y_pred")
        plt.set(xlim=(-7, 2))
        plt.set(ylim=(-7, 2))
        plt

    
    mean_mse = np.mean([x.cpu() for x in mse_losses])
    mean_l1 = np.mean([x.cpu() for x in l1_losses])

    return mean_mse, mean_l1, dfs




def predict_classification(model, test_loader, device, best_model_weights=None, plot_final = True, plot_label="precision-recall curve"):

    dfs = []

    test_true_labels = []
    test_predicted_labels = []

    #load best model
    if best_model_weights is not None:
        print("best weights loaded")
        model.load_state_dict(best_model_weights)

    for test_batch in test_loader:
        with torch.no_grad():
            test_true_labels.extend(test_batch.y.int().tolist())


            test_batch.to(device)
            pred, embed = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch) 
            df = pd.DataFrame()
            df["y_real"] = test_batch.y.tolist()
            df["y_pred"] = pred.tolist()

            df["y_real"] = df["y_real"].apply(lambda row: row[0])
            df["y_pred"] = df["y_pred"].apply(lambda row: row[0])
            dfs.append(df)


            test_predicted_labels.extend(pred.tolist())


    precisions, recalls, thresholds = precision_recall_curve(np.array(test_true_labels).flatten(), np.array(test_predicted_labels).flatten(), drop_intermediate=True)

    if plot_final:
        plt.plot(recalls, precisions)
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.title(plot_label)


    return precisions, recalls, thresholds, dfs




