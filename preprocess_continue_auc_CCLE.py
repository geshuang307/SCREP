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
import sys
sys.path.append('./')
sys.path.append('../')
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
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
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

    ############### ccle ##########
    reader = csv.reader(open(folder + "drug_smiles_ccle1448_own.csv"))
    ###############################
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



def save_cell_oge_matrix():
    
    ############## CCLE gene expression file ##################
    file_path = "/workspace/geshuang/data/CCLE/CCLE_expression.csv"
    df = pd.read_csv(file_path, sep = ',')
    gene_expression = df.iloc[:, 1:]
    gene_expression = np.array(gene_expression)
    gene_names = df.columns[1:]
    gene_names = [gene_names[i].split(' (')[0] for i in range(len(gene_names))]
    panglao = sc.read_h5ad('data/panglao_10000.h5ad')
    ref = panglao.var_names.tolist()
    obj = gene_names
    counts = np.zeros((gene_expression.shape[0], panglao.X.shape[1]))     # Force alignment to 16,906 feature dimensions, initialize to zero
    number_of_genes = 0
    for i in range(len(ref)):
        if ref[i] in obj:
            gene = obj.index(ref[i])   # find the index of gene 'ref[i]' in obj
            counts[:,i] = gene_expression[:, gene]
            number_of_genes += 1
    print('total gene numbers:', number_of_genes)        # 16859
    cell_names = df.iloc[:,0].tolist()
    cell_features = counts
    max = np.max(cell_features)
    min = np.min(cell_features)
    # print(cell_features.shape)
    cell_dict = {}
    for index in range(len(cell_names)):
        cell_name = cell_names[index]
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
    path = "/workspace/geshuang/data/CCLE/secondary-screen-dose-response-curve-parameters.csv"
    f = open(path)
    df = pd.read_csv(path)
    auc = df.iloc[:, 8].tolist()
    # z_score = (auc-np.mean(auc))/np.var(auc)
    max = np.max(auc)
    min = np.min(auc)
    reader = csv.reader(f)
    next(reader)


    cell_dict_ge, cell_feature_ge, ref = save_cell_oge_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()

    temp_data = []

    ########## 根据z分数定义敏感和耐药的标签 ####################

    num_res, num_sens = 0, 0
    iter = 0
    for item in reader:
        
        drug = item[11]
        cell = item[1]

        auc1 = item[8]
        auc_new = (float(auc1)-min)/(max-min)
        # z_score = 8
        # ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        # ec50 = 1 / (1 + pow(math.exp(float(ec50)), -0.1))
        
        # if float(z_score[iter]) >= 1.65:      # 1.65
        #     binary_auc = 0          # resistant
        #     temp_data.append((drug, cell, int(binary_auc)))  
        #     num_res += 1
        # elif float(z_score[iter]) <= -1.65:    # -1.65
        #     binary_auc = 1          # sensitive
        #     temp_data.append((drug, cell, int(binary_auc)))  
        #     num_sens += 1
        # else:
        #     pass
        iter += 1
        # temp_data.append((drug, cell, ic50))                    # 224510
        temp_data.append((drug, cell, auc_new))    
    print(len(temp_data))     # 19601 
    # print(num_res)            # 8614
    # print(num_sens)           # 10987
    xd = []
    xc_ge = []
    y = []
    lst_drug = []
    lst_cell = []
    random.seed(2024)
    random.shuffle(temp_data)            
########################## whether to use PCA ##################
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
        cell_feature_ge = mmscaler.fit_transform(cell_feature_ge)              # 对bulk数据做归一化
        
#################################################################
    for data in temp_data:
        drug, cell, auc = data
        # if drug in drug_dict and cell in cell_dict_ge and cell in cell_dict_meth and drug != 'Erlotinib':
        if drug in drug_dict and cell in cell_dict_ge:
            xd.append(drug_smile[drug_dict[drug]])
            xc_ge.append(cell_feature_ge[cell_dict_ge[cell]])

            y.append(auc)
            
            lst_drug.append(drug)
            lst_cell.append(cell)

    # with open('drug_dict_CCLE', 'wb') as fp:
    #     pickle.dump(drug_dict, fp)
    with open('drug_cell_list/CCLE/all_cell_list_CCLE.txt', 'w') as file:
        for item in lst_cell:
            file.write(item + '\n')
    with open('drug_cell_list/CCLE/all_drug_list_CCLE.txt', 'w') as file:
        for item in lst_drug:
            file.write(item + '\n')

    xd_all = np.asarray(xd)              # 147430
    xc_ge_all = np.asarray(xc_ge)
    y_all = np.asarray(y)

    xd_all_trainval, xd_test, xc_ge_all_trainval, xc_ge_test, y_all_trainval, y_test = train_test_split(xd_all, xc_ge_all, y_all, test_size=0.1, random_state=42)     # 划分训练+验证/测试集
    xd_train, xd_val, xc_ge_train, xc_ge_val, y_train, y_val = train_test_split(xd_all_trainval, xc_ge_all_trainval, y_all_trainval, test_size=1/9, random_state=42)  
    # print('train sample number', xd_train.shape)           # 95920   14862
    # print('val sample number', xd_val.shape)               # 11990   1662
    # print('test sample number', xd_test.shape)             # 11991   1662
    # rds = RandomOverSampler(random_state=2024)
    # print('origion samples of sensitive: ', sum(y_train))    # 11375->135376
    # oversample = list(zip(xd_train, sc_ge_train))
    # oversample = np.array(oversample)
    # oversample, y_train = rds.fit_resample(oversample, y_train)    # 135376
    # print('number of sensitive samples after oversample: ', sum(y_train))   # 84545
    # print('total number of samples after oversample: ', y_train.shape[0])      # 169090
    # xd_train, xc_ge_train = zip(*oversample)
    # xd_train = np.array(xd_train)
    # xc_ge_train = np.array(xc_ge_train)
    dataset = 'CCLE_maxmin'
    # print('preparing ', dataset + '_train.pt in pytorch format!')

    # xd_all_1, xc_ge_all_1, y_all_1, xd_all_0, xc_ge_all_0, y_all_0 = [], [] , [], [], [], []
    # for index in range(y_all.shape[0]):
    #     if y_all[index] == 0:
    #         xd_all_0.append(xd_all[index])
    #         xc_ge_all_0.append(xc_ge_all[index])
    #         y_all_0.append(y_all[index])
    #     elif y_all[index] == 1:
    #         xd_all_1.append(xd_all[index])
    #         xc_ge_all_1.append(xc_ge_all[index])
    #         y_all_1.append(y_all[index])
    #     else:
    #         pass

    all_data = TestbedDataset(root='data', dataset=dataset+'_all_auc_gene16906_index', xd=xd, xt_ge=xc_ge, xt_meth=None, xt_mut=None, y=y, smile_graph=smile_graph)
    train_data = TestbedDataset(root='data', dataset=dataset+'_train_auc_gene16906_all', xd=xd_train, xt_ge=xc_ge_train, xt_meth=None, xt_mut=None, y=y_train, smile_graph=smile_graph)   # Erlotinib: 152530 ALL:152827 
    val_data = TestbedDataset(root='data', dataset=dataset+'_val_auc_gene16906_all', xd=xd_val, xt_ge=xc_ge_val, xt_meth=None,xt_mut=None, y=y_val, smile_graph=smile_graph)              # Erlotinib:19066   ALL:19103
    test_data = TestbedDataset(root='data', dataset=dataset+'_test_auc_gene16906_all', xd=xd_test, xt_ge=xc_ge_test, xt_meth=None, xt_mut=None, y=y_test, smile_graph=smile_graph)        # 19067
    # all_data0 = TestbedDataset(root = 'data', dataset = dataset + '_allclass0_z_score_auc_gene16906', xd = xd_all_0, xt_ge = xc_ge_all_0, xt_meth=None, xt_mut=None, y = y_all_0, smile_graph = smile_graph)   # 7335
    # all_data1 = TestbedDataset(root = 'data', dataset = dataset + '_allclass1_z_score_auc_gene16906', xd = xd_all_1, xt_ge = xc_ge_all_1, xt_meth=None, xt_mut=None,y = y_all_1, smile_graph = smile_graph)   # 9277
 
 
    print("build data complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='prepare dataset to train model')
    parser.add_argument('--choice', type=int, required=False, default=3, help='0.KernelPCA, 1.PCA, 2.Isomap')
    args = parser.parse_args()
    choice = args.choice
    save_mix_drug_cell_matrix(choice)