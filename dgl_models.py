import torch
from torch.nn import Linear
import torch.nn.functional as F 

from copy import deepcopy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import *

from dgllife.model import GCNPredictor

from torch_geometric.utils.convert import * 

from custom_general_functions import *


#def initialize_dgl_regression_model(in_feats, hidden_feats=None, gnn_norm=None, activation=None, residual=None, batchnorm=None, dropout=None, classifier_hidden_feats=128, classifier_dropout=0.0, n_tasks=1, predictor_hidden_feats=128, predictor_dropout=0.0, learning_rate=0.001):
def initialize_dgl_regression_model(in_feats, hidden_feats=None, gnn_norm=None, activation=None, residual=None, batchnorm=None, dropout=None, classifier_dropout=0.0, n_tasks=1, predictor_hidden_feats=128, predictor_dropout=0.0, learning_rate=0.001):



    #model = GCNPredictor(in_feats=in_feats, hidden_feats=hidden_feats, gnn_norm=gnn_norm, activation=activation, residual=residual, batchnorm=batchnorm, dropout=dropout, classifier_hidden_feats=classifier_hidden_feats, classifier_dropout=classifier_dropout, n_tasks=n_tasks, predictor_hidden_feats=predictor_hidden_feats, predictor_dropout=predictor_dropout)
    model = GCNPredictor(in_feats=in_feats, hidden_feats=hidden_feats, gnn_norm=gnn_norm, activation=activation, residual=residual, batchnorm=batchnorm, dropout=dropout, classifier_dropout=classifier_dropout, n_tasks=n_tasks, predictor_hidden_feats=predictor_hidden_feats, predictor_dropout=predictor_dropout)

    
    print(model)
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    loss_fn = torch.nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, device, optimizer, loss_fn




def initialize_dgl_classification_model(in_feats, hidden_feats=None, gnn_norm=None, activation=None, residual=None, batchnorm=None, dropout=None, classifier_hidden_feats=128, classifier_dropout=0.0, n_tasks=1, predictor_hidden_feats=128, predictor_dropout=0.0, learning_rate=0.001):


    model = GCNPredictor(in_feats=in_feats, hidden_feats=hidden_feats, gnn_norm=gnn_norm, activation=activation, residual=residual, batchnorm=batchnorm, dropout=dropout, classifier_hidden_feats=classifier_hidden_feats, classifier_dropout=classifier_dropout, n_tasks=n_tasks, predictor_hidden_feats=predictor_hidden_feats, predictor_dropout=predictor_dropout)
    
    
    print(model)
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, device, optimizer, loss_fn



def regression_dgl_train(model, loader, test_loader, device, loss_fn, optimizer, log_time=10, max_epochs=1000, apply_early_stopping = True, early_stopping_patience = 50, finally_plot_losses = False):
    
    model.train()
    losses = []
    val_losses = []

    best_loss = float('inf')
    best_model_weights = None

    model = model.to(device)

    if apply_early_stopping:
        patience = early_stopping_patience

    for epoch in range(max_epochs):
        running_loss = 0
        running_val_loss=0
        
        model.train()


        for batch_dgl, labels in loader:
            #new_y = batch.y
            #batch_dgl = to_dgl(batch)
            labels = labels.float()
            labels = labels.unsqueeze(dim=1)
            labels = labels.to(device)
            
            batch_dgl_data = batch_dgl.ndata['h'].float()
            

            optimizer.zero_grad() 

            batch_dgl = batch_dgl.to(device)  
            batch_dgl_data = batch_dgl_data.to(device)
            pred= model(batch_dgl, batch_dgl_data) 


            loss = loss_fn(pred, labels)     
            loss.backward()  

            optimizer.step() 

            running_loss += loss.item()

        model.eval()
        for test_batch, test_labels in test_loader:
            test_labels = test_labels.float()
            test_labels = test_labels.unsqueeze(dim=1)
            test_labels = test_labels.to(device)
            test_batch = test_batch.to(device)  
            
            test_batch_dgl_data = test_batch.ndata['h'].float()
            pred= model(test_batch, test_batch_dgl_data) 
            val_loss = loss_fn(pred, test_labels)     

            running_val_loss += val_loss.item()
        
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
                    print("patience 0")
                    break
    
    
    
    if finally_plot_losses:
        plot_losses(losses, val_losses, log_time)

    return model, best_model_weights, losses, val_losses




def predict_dgl_regression(model, test_loader, device, best_model_weights = None, plot_final = True, return_df=True):

    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()

    #load best model
    if best_model_weights is not None:
        print("best weights loaded")
        model.load_state_dict(best_model_weights)

    mse_losses = []
    l1_losses = []
    dfs = []
    for test_batch, test_labels in test_loader:
        with torch.no_grad():
            test_labels = test_labels.float()
            test_labels = test_labels.unsqueeze(dim=1)
            test_labels = test_labels.to(device)
            test_batch = test_batch.to(device)  
            test_batch_dgl_data = test_batch.ndata['h'].float()

            pred= model(test_batch, test_batch_dgl_data) 
            #test_batch.to(device)
            
            df = pd.DataFrame()
            df["y_real"] = test_labels.tolist()
            df["y_pred"] = pred.tolist()

            df["y_real"] = df["y_real"].apply(lambda row: row[0])
            df["y_pred"] = df["y_pred"].apply(lambda row: row[0])
            dfs.append(df)

            mse_losses.append(mse_loss(pred, test_labels).item())
            l1_losses.append(l1_loss(pred,test_labels).item())

    final_df = pd.concat(dfs)

    if plot_final:
        plt = sns.scatterplot(data=final_df, x="y_real", y="y_pred")
        plt.set(xlim=(-7, 2))
        plt.set(ylim=(-7, 2))
        plt

    mean_mse = np.mean(mse_losses)
    mean_l1 = np.mean(l1_losses)

    if not return_df:
        return mean_mse, mean_l1
    else:
        return mean_mse, mean_l1, dfs
    









def classification_dgl_train(model, loader, test_loader, device, loss_fn, optimizer, log_time=10, max_epochs=1000, apply_early_stopping = True, early_stopping_patience = 50, finally_plot_losses = False):
    from scipy.special import expit
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix, accuracy_score

    model.train()
    losses = []
    val_losses = []

    best_loss = float('inf')
    best_model_weights = None

    model = model.to(device)

    if apply_early_stopping:
        patience = early_stopping_patience

    for epoch in range(max_epochs):
        val_accuracies = []

        running_loss = 0
        running_val_loss=0
        
        model.train()


        for batch_dgl, labels in loader:

            #batch_dgl = to_dgl(batch)
            labels = labels.float()
            labels = labels.unsqueeze(dim=1)
            labels = labels.to(device)
            
            batch_dgl_data = batch_dgl.ndata['h'].float()
            

            optimizer.zero_grad() 

            batch_dgl = batch_dgl.to(device)  
            batch_dgl_data = batch_dgl_data.to(device)
            pred= model(batch_dgl, batch_dgl_data) 

            y_pred = np.round(expit(pred.detach().cpu().numpy().flatten()))


            loss = loss_fn(pred, labels)     
            loss.backward()  

            optimizer.step() 

            running_loss += loss.item()

        model.eval()
        predicted_labels = []
        true_labels = []
        #correct = 0
        #total = 0
        for test_batch, test_labels in test_loader:
            test_labels = test_labels.float()
            test_labels = test_labels.unsqueeze(dim=1)
            test_labels = test_labels.to(device)
            test_batch = test_batch.to(device)  
            
            test_batch_dgl_data = test_batch.ndata['h'].float()
            pred= model(test_batch, test_batch_dgl_data) 

            y_pred = np.round(expit(pred.detach().cpu().numpy().flatten()))

            val_accuracies.append(accuracy_score(test_labels.tolist(), y_pred.flatten()))

            val_loss = loss_fn(pred, test_labels)     

            running_val_loss += val_loss.item()

            true_labels.extend(test_labels.tolist())
            predicted_labels.extend(y_pred)

        accuracy = np.mean(val_accuracies)

        #avg_val_loss = np.mean(val_losses)
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
                    print("patience 0")
                    break
        
    
    if finally_plot_losses:
        plot_losses(losses, val_losses, log_time)

    return model, best_model_weights, losses, val_losses




def predict_classification(model, test_loader, device, plot_final = True, plot_label="precision-recall curve"):

    dfs = []

    test_true_labels = []
    test_predicted_labels = []
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
        plt.plot(recalls, precisions, label=plot_label)


    return precisions, recalls, thresholds, dfs




def predict_dgl_classification(model, test_loader, device, best_model_weights = None, plot_final = True, plot_label="precision-recall curve"):


    dfs = []

    #load best model
    if best_model_weights is not None:
        print("best weights loaded")
        model.load_state_dict(best_model_weights)

    test_true_labels = []
    test_predicted_labels = []
    for test_batch, test_labels in test_loader:
        with torch.no_grad():
            test_true_labels.extend(test_labels)

            test_labels = test_labels.float()
            test_labels = test_labels.unsqueeze(dim=1)
            test_labels = test_labels.to(device)
            test_batch = test_batch.to(device)  
            test_batch_dgl_data = test_batch.ndata['h'].float()

            pred= model(test_batch, test_batch_dgl_data) 
            #test_batch.to(device)
            #pred, embed = model(test_batch.x.float(), test_batch.edge_index, test_batch.batch) 
            df = pd.DataFrame()
            df["y_real"] = test_labels.tolist()
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



