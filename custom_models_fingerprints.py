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



class FPN(torch.nn.Module):
    def __init__(self, linear_layers = [1024, 512]):
        super(FPN, self).__init__()

        self.nns = torch.nn.ModuleList()

        for i, size in enumerate(linear_layers):
            self.nns.append(torch.nn.LazyLinear(size))


        self.out = torch.nn.LazyLinear(1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):

        for lnn in self.nns:
            x = self.relu(lnn(x))

        x = self.out(x)

        return x



#currently, this function is identical to regression
def initialize_classification_model_fingerprint(linear_layers=[512], learning_rate=0.001):

    model = FPN(linear_layers=linear_layers)
    #print(model)
    #print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, device, optimizer, loss_fn



def initialize_regression_model_fingerprint(linear_layers=[512], learning_rate=0.001):

    model = FPN(linear_layers=linear_layers)
    #print(model)
    #print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    loss_fn = torch.nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, device, optimizer, loss_fn




def regression_train_fingerprint(model, loader, test_loader, device, loss_fn, optimizer, log_time=10, max_epochs=1000, apply_early_stopping = True, early_stopping_patience = 50, finally_plot_losses = False):
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

            #batch.to(device)  
            batch[-1] = batch[-1].to(device)
            batch[-1] = batch[-1].squeeze(dim=-1)
            
            optimizer.zero_grad() 

            #pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch) 
            pred = model(batch[0].to(device) ) 

            #loss = loss_fn(pred, batch.y)  
            loss = loss_fn(pred, batch[-1].to(device) )   

            loss.backward()  

            optimizer.step() 

            running_loss += loss.item()

        model.eval()
        for test_batch in test_loader:
            test_batch[-1] = test_batch[-1].to(device)
            test_batch[-1] = test_batch[-1].squeeze(dim=-1)
            #test_batch.to(device)  

            #pred, embedding = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch) 
            pred = model(test_batch[0].to(device) ) 

            #val_loss = loss_fn(pred, test_batch.y)    
            val_loss = loss_fn(pred, test_batch[-1].to(device))    

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
    plt.show()
    return model, best_model_weights, losses, val_losses


def predict_regression_fingerprint(model, test_loader, device, best_model_weights=None, plot_final = True):
    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()

    mse_losses = []
    l1_losses = []
    dfs = []

    if best_model_weights is not None:
        print("best weights loaded")
        model.load_state_dict(best_model_weights)


    for test_batch in test_loader:
        with torch.no_grad():
            
            #print("this is test_batch[1] FIRST")
            #print(test_batch[1])

            test_batch[0] = test_batch[0].to(device)
            test_batch[-1] = test_batch[-1].to(device)
            test_batch[-1] = test_batch[-1].squeeze(dim=-1)
            #test_batch.to(device)
            #pred, embed = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch) 

                       
            #print("this is test_batch[1] SECOND")
            #print(test_batch[1])
            pred = model(test_batch[0].to(device)) 


                                   
           
            df = pd.DataFrame()
            df["y_real"] = test_batch[-1].tolist()
            df["y_pred"] = pred.tolist()

            df["y_real"] = df["y_real"].apply(lambda row: row[0])
            df["y_pred"] = df["y_pred"].apply(lambda row: row[0])
            dfs.append(df)

 

            mse_losses.append(mse_loss(pred, test_batch[-1]).item())
            l1_losses.append(l1_loss(pred, test_batch[-1]).item())

    final_df = pd.concat(dfs)

    if plot_final:
        plt = sns.scatterplot(data=final_df, x="y_real", y="y_pred")
        plt.set(xlim=(-7, 2))
        plt.set(ylim=(-7, 2))
        plt

    
    mean_mse = np.mean([x for x in mse_losses])
    mean_l1 = np.mean([x for x in l1_losses])

    return mean_mse, mean_l1, dfs


def classification_train_fingerprint(model, loader, test_loader, device, loss_fn, optimizer, log_time=10, max_epochs=1000, apply_early_stopping = True, early_stopping_patience = 50, finally_plot_losses = False):
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
            batch[-1] = batch[-1].to(device)
            batch[-1] = batch[-1].squeeze(dim=-1)
        #batch.y = batch.y.float()
        #batch.y = batch.y.squeeze()

            ##batch[0].to(device)  
            ##batch[-1].to(device)

            #batch.to(device)  

            optimizer.zero_grad() 

            #pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch) 
            pred = model(batch[0].to(device) ) 

            #y_pred = np.round(expit(pred.detach().cpu().numpy().flatten()))


            loss = loss_fn(pred, batch[-1].to(device) )       
            loss.backward()  

            optimizer.step() 

            running_loss += loss.item()


        model.eval()

        predicted_labels = []
        true_labels = []
        #correct = 0
        #total = 0
        for test_batch in test_loader:
            test_batch[-1] = test_batch[-1].to(device)
            test_batch[-1] = test_batch[-1].squeeze(dim=-1)
            
            #test_batch.to(device)  

            #pred, embedding = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch) 
            pred = model(test_batch[0].to(device) ) 

            y_pred = np.round(expit(pred.detach().cpu().numpy().flatten()))

            val_accuracies.append(accuracy_score(test_batch[1].tolist(), y_pred.flatten()))
            #val_accuracies.append(accuracy_score(test_batch.y.tolist(), y_pred.flatten()))

            #val_loss = loss_fn(pred, test_batch.y)     
            val_loss = loss_fn(pred, test_batch[1].to(device))  

            running_val_loss += val_loss.item()
            #total += test_batch.y.size(0)
            #_, predicted = torch.max(pred, 1)
            #correct += (predicted == test_batch.y).sum().item()

            true_labels.extend(test_batch[1])
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





def predict_classification_fingerprint(model, test_loader, device, best_model_weights=None, plot_final = True, plot_label="precision-recall curve"):

    dfs = []

    test_true_labels = []
    test_predicted_labels = []

    if best_model_weights is not None:
        print("best weights loaded")
        model.load_state_dict(best_model_weights)

    for test_batch in test_loader:
        with torch.no_grad():

            test_batch[0] = test_batch[0].to(device)
            test_batch[-1] = test_batch[-1].to(device)
            test_batch[-1] = test_batch[-1].squeeze(dim=-1)

            test_true_labels.extend(test_batch[1].tolist())

            #test_batch.to(device)
            #pred, embed = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch) 
            pred = model(test_batch[0] ) 

            df = pd.DataFrame()
            df["y_real"] = test_batch[1].tolist()#.y.tolist()
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
    

