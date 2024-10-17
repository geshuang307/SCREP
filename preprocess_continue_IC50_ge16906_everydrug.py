import os
import csv
from pubchempy import *
import numpy as np
import numbers
import h5py
import math
import pandas as pd
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from torch._C import device
from utils_sc import *
import random
import pickle
import sys
import matplotlib.pyplot as plt
import argparse
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import  Isomap
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import scanpy as sc, numpy as np, pandas as pd, anndata as ad
from sklearn import preprocessing
def is_not_float(string_list):
    try:
        for string in string_list:
            float(string)
        return False
    except:
        return True

"""
The following 4 function is used to preprocess the drug data. We download the drug list manually, and download the SMILES format using pubchempy. Since this part is time consuming, I write the cids and SMILES into a csv file. 
"""

folder = "global_data/"
#folder = ""

def load_drug_list():
    filename = folder + "Druglist.csv"
    csvfile = open(filename, "rb")
    reader = csv.reader(csvfile)
    next(reader, None)
    drugs = []
    for line in reader:
        drugs.append(line[0])
    drugs = list(set(drugs))
    return drugs

def write_drug_cid():
    drugs = load_drug_list()
    drug_id = []
    datas = []
    outputfile = open(folder + 'pychem_cid.csv', 'wb')
    wr = csv.writer(outputfile)
    unknow_drug = []
    for drug in drugs:
        c = get_compounds(drug, 'name')
        if drug.isdigit():
            cid = int(drug)
        elif len(c) == 0:
            unknow_drug.append(drug)
            continue
        else:
            cid = c[0].cid
        print(drug, cid)
        drug_id.append(cid)
        row = [drug, str(cid)]
        wr.writerow(row)
    outputfile.close()
    outputfile = open(folder + "unknow_drug_by_pychem.csv", 'wb')
    wr = csv.writer(outputfile)
    wr.writerow(unknow_drug)

def cid_from_other_source():
    """
    some drug can not be found in pychem, so I try to find some cid manually.
    the small_molecule.csv is downloaded from http://lincs.hms.harvard.edu/db/sm/
    """
    f = open(folder + "small_molecule.csv", 'r')
    reader = csv.reader(f)
    reader.next()
    cid_dict = {}
    for item in reader:
        name = item[1]
        cid = item[4]
        if not name in cid_dict: 
            cid_dict[name] = str(cid)

    unknow_drug = open(folder + "unknow_drug_by_pychem.csv").readline().split(",")
    drug_cid_dict = {k:v for k,v in cid_dict.iteritems() if k in unknow_drug and not is_not_float([v])}
    return drug_cid_dict

def load_cid_dict():
    reader = csv.reader(open(folder + "pychem_cid.csv"))
    pychem_dict = {}
    for item in reader:
        pychem_dict[item[0]] = item[1]
    pychem_dict.update(cid_from_other_source())
    return pychem_dict


def download_smiles():
    cids_dict = load_cid_dict()
    cids = [v for k,v in cids_dict.iteritems()]
    inv_cids_dict = {v:k for k,v in cids_dict.iteritems()}
    download('CSV', folder + 'drug_smiles.csv', cids, operation='property/CanonicalSMILES,IsomericSMILES', overwrite=True)
    f = open(folder + 'drug_smiles.csv')
    reader = csv.reader(f)
    header = ['name'] + reader.next()
    content = []
    for line in reader:
        content.append([inv_cids_dict[line[0]]] + line)
    f.close()
    f = open(folder + "drug_smiles.csv", "w")
    writer = csv.writer(f)
    writer.writerow(header)
    for item in content:
        writer.writerow(item)
    f.close()

"""
The following code will convert the SMILES format into onehot format
"""

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index

def load_drug_smile():
    #  GDSC
    reader = csv.reader(open(folder + "drug_smiles.csv"))
    next(reader, None)

    drug_dict = {}
    drug_smile = []

    for item in reader:
        name = item[0]
        smile = item[2]

        if name in drug_dict:
            pos = drug_dict[name]
        else:
            pos = len(drug_dict)
            drug_dict[name] = pos
        drug_smile.append(smile)
    
    smile_graph = {}
    for smile in drug_smile:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    
    return drug_dict, drug_smile, smile_graph


"""
This part is used to read PANCANCER Gene Expression Cell line features
"""

def save_cell_ge_matrix():
    f = open(folder + "Cell_line_RMA_proc_basalExp.csv")
    reader = csv.reader(f)
    firstRow = next(reader)
    numberCol = len(firstRow) - 1
    features = {}
    cell_dict = {}
    matrix_list = []
    for item in reader:
        cell_id = item[0]
        ge = []
        for i in range(1, len(item)):
            ge.append(int(item[i]))
        cell_dict[cell_id] = np.asarray(ge)
    return cell_dict


def save_cell_oge_matrix():
    file_path = folder + "Cell_line_RMA_proc_basalExp.txt"
    df = pd.read_csv(file_path, sep = '\t')
    print(df)
    gene_expression = df.iloc[0:, 2:]
    gene_expression = gene_expression.T
    gene_expression = np.array(gene_expression)
    gene_names = df.iloc[:,0]

    panglao = sc.read_h5ad('data/panglao_10000.h5ad')
    ref = panglao.var_names.tolist()
    obj = gene_names.tolist()

    counts = np.zeros((gene_expression.shape[0], panglao.X.shape[1]))     # Forcefully align the gene count to a dimension of 16,906.

    for i in range(len(ref)):
        if ref[i] in obj:
            gene = obj.index(ref[i])   # To find the index of an object named ref[i] in a list or array of gene names
            counts[:,i] = gene_expression[:, gene]
    
    cell_names = df.columns[2:].tolist()
    cell_features = counts
    max = np.max(cell_features)
    min = np.min(cell_features)
    print(cell_features.shape)
    cell_dict = {}
    for index in range(len(cell_names)):
        cell_name = cell_names[index].replace('DATA.', '')
        cell_dict[cell_name] = []
        cell_feature = (cell_features[index, :]-min)/(max-min)
        cell_dict[cell_name].append(cell_feature)
    i = 0
    for cell in list(cell_dict.keys()):
        cell_dict[cell] = i
        i += 1
    return cell_dict, cell_features, ref

def train(model, device, train_loader, optimizer, epoch, log_interval, model_st):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    loss_ae = nn.MSELoss()
    avg_loss = []
    weight_fn = 0.01
    weight_ae = 2
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        #For non-variational autoencoder
        if 'VAE' not in model_st:
            output, _ = model(data)
            loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        else:
        #For variation autoencoder
            output, _, decode, log_var, mu = model(data)
            loss = weight_fn*loss_fn(output, data.y.view(-1, 1).float().to(device)) + loss_ae(decode, data.target_mut[:,None,:].float().to(device)) + torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    return sum(avg_loss)/len(avg_loss)

def predicting(model, device, loader, model_st):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            #Non-variational autoencoder
            if 'VAE' not in model_st:
                output, _ = model(data)
            else:
            #Variational autoencoder
                output, _, decode, log_var, mu = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


"""
This part is used to extract the drug - cell interaction strength. it contains IC50, AUC, Max conc, RMSE, Z_score
"""
class DataBuilder(Dataset):
    def __init__(self, cell_feature_ge):
        self.cell_feature_ge = cell_feature_ge
        self.cell_feature_ge = torch.FloatTensor(self.cell_feature_ge)
        self.len = self.cell_feature_ge[0]
    
    def __getitem__(self, index):
        return self.cell_feature_ge[index]

    def __len__(self):
        return self.len

def save_mix_drug_cell_matrix(choice):
    f = open(folder + "PANCANCER_IC.csv")
    reader = csv.reader(f)
    next(reader)

    cell_dict_ge, cell_feature_ge, ref = save_cell_oge_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()

    temp_data = []
    for item in reader:
        drug = item[0]
        cell = item[3]
        ic50 = item[8]
        ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        temp_data.append((drug, cell, ic50))                    # 224510

    xd = []
    xc_ge = []
    y = []
    lst_drug = []
    lst_cell = []
    random.seed(2024)
    random.shuffle(temp_data)  
    ################################# 
    import scanpypip.preprocessing as pp
    import seaborn as sns
    from matplotlib import pyplot as plt
######################### For visualization ###############################################
    # new = sc.AnnData(X = cell_feature_ge)
    # adata = pp.cal_ncount_ngenes(new)      # compute gene counts of each cell
    # sns.histplot(adata.obs["total_counts"], bins=100, kde=False)
    # plt.xticks(fontproperties = 'Arial', size = 14)
    # plt.yticks(fontproperties = 'Arial', size = 14)
    # plt.savefig('histogram.png', dpi=500)             
########################## choose wheater to use PCA to reduce dimensions ##################
    if choice == 0:
        # Kernel PCA
        kpca = KernelPCA(n_components=1000, kernel='rbf', gamma=131, random_state=42)
        cell_feature_ge = kpca.fit_transform(cell_feature_ge)
    elif choice == 1:
        # PCA
        pca = PCA(n_components=1000)
        cell_feature_ge = pca.fit_transform(cell_feature_ge)
    elif choice == 2:
        #Isomap
        isomap = Isomap(n_components=480)
        cell_feature_ge = isomap.fit_transform(cell_feature_ge)
    else:
        mmscaler = preprocessing.MinMaxScaler()
        cell_feature_ge = mmscaler.fit_transform(cell_feature_ge)  # normalization in bulk data
        # pass
################## For visualization: Normalization ###############################
    # new = sc.AnnData(X = cell_feature_ge)
    # adata = pp.cal_ncount_ngenes(new)      
    # sns.histplot(adata.obs["total_counts"], bins=100, kde=False)
    # plt.xticks(fontproperties = 'Arial', size = 14)
    # plt.yticks(fontproperties = 'Arial', size = 14)
    # plt.savefig('histogram_normalize.png', dpi=500)
#################################################################
    for data in temp_data:
        drug, cell, ic50 = data
        # if drug in drug_dict and cell in cell_dict_ge and cell in cell_dict_meth and drug != 'Erlotinib':
        if drug in drug_dict and cell in cell_dict_ge:
            xd.append(drug_smile[drug_dict[drug]])
            xc_ge.append(cell_feature_ge[cell_dict_ge[cell]])
            y.append(ic50)        
            lst_drug.append(drug)
            lst_cell.append(cell)

    with open('data/drug_dict', 'wb') as fp:
        pickle.dump(drug_dict, fp)
    with open('drug_cell_list/all_cell_list.txt', 'w') as file:
        for item in lst_cell:
            file.write(item + '\n')
    with open('drug_cell_list/all_drug_list.txt', 'w') as file:
        for item in lst_drug:
            file.write(item + '\n')
    xd = np.asarray(xd)              # 147430
    xc_ge = np.asarray(xc_ge)
    y = np.asarray(y)

    ################# For visualization: Save bulk data in AnnData format and set the bulk label to 0. ################

    # new = ad.AnnData(X = cell_feature_ge)
    # sc.tl.pca(new,svd_solver='arpack')
    # sc.pp.neighbors(new, n_neighbors=10)
    # # Generate cluster labels   
    # sc.tl.leiden(new, resolution=0.2)
    # sc.tl.umap(new)
    # new.obs['leiden_origin']= new.obs['leiden']
    # new.obsm['X_umap_origin']= new.obsm['X_umap']
    # new.var_names = ref
    # new.obs['bulk_sc'] = [0] * cell_feature_ge.shape[0]
    # new.write("save/"+ 'GDSC_norm' +"_new.h5ad")
    ###########################################################

    # size = int(xd.shape[0] * 0.8)
    # size1 = int(xd.shape[0] * 0.9)

    # np.save('data/list_drug_mix_test', lst_drug[size1:])
    # np.save('data/list_cell_mix_test', lst_cell[size1:])
    # with open('data/list_drug_mix_test', 'wb') as fp:
    #     pickle.dump(lst_drug[size1:], fp)
        
    # with open('data/list_cell_mix_test', 'wb') as fp:
    #     pickle.dump(lst_cell[size1:], fp)

    # xd_train = xd[:size]               # 191034
    # xd_val = xd[size:size1]
    # xd_test = xd[size1:]

    # xc_ge_train = xc_ge[:size]
    # xc_ge_val = xc_ge[size:size1]
    # xc_ge_test = xc_ge[size1:]
  
    # y_train = y[:size]
    # y_val = y[size:size1]
    # y_test = y[size1:]

    dataset = 'GDSC_norm'
    print('preparing ', dataset + '_train.pt in pytorch format!')

    all_data = TestbedDataset(root='data', dataset=dataset+'_all_continueic50_gene16906_index', xd=xd, xt_ge=xc_ge, xt_meth=None, xt_mut=None, y=y, smile_graph=smile_graph) 
    # train_data = TestbedDataset(root='data', dataset=dataset+'_train_continueic50_gene16906_all', xd=xd_train, xt_ge=xc_ge_train, xt_meth=None, xt_mut=None, y=y_train, smile_graph=smile_graph)   # Erlotinib: 152530 ALL:152827 
    # val_data = TestbedDataset(root='data', dataset=dataset+'_val_continueic50_gene16906_all', xd=xd_val, xt_ge=xc_ge_val, xt_meth=None,xt_mut=None, y=y_val, smile_graph=smile_graph)              # Erlotinib:19066   ALL:19103
    # test_data = TestbedDataset(root='data', dataset=dataset+'_test_continueic50_gene16906_all', xd=xd_test, xt_ge=xc_ge_test, xt_meth=None, xt_mut=None, y=y_test, smile_graph=smile_graph)        # 19067
 
    print("build data complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='prepare dataset to train model')
    parser.add_argument('--choice', type=int, required=False, default=3, help='0.KernelPCA, 1.PCA, 2.Isomap')
    args = parser.parse_args()
    choice = args.choice
    save_mix_drug_cell_matrix(choice)