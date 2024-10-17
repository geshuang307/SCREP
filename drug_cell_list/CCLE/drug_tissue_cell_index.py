import csv
import json
import pandas as pd

with open('drug_cell_list/CCLE/tissue_type.json', 'r') as file:   # read all tissue types
    tissue_dict = json.load(file)
with open('drug_cell_list/CCLE/cell_tissue.json', 'r') as file:   # read "cell names：tissue" dict
    csv_dict = json.load(file)
drug = 'all'
cell_index_list = []
with open('drug_cell_list/CCLE/' + drug + '_cell_list_CCLE.txt', 'r') as f:      # Read the cell names from the dataset and arrange them in a list according to the order in the dataset.
    for line in f:
        cell_index_list.append(line.strip())

drug_index_list = []
with open('drug_cell_list/CCLE/' + drug + '_drug_list_CCLE.txt', 'r') as f:
    for line in f:
        drug_index_list.append(line.strip())
        
all_drug = list(set(drug_index_list))              # 1434
drug_tissue_cell_index = {}
for drug in all_drug:
    drug_index = [i for i, value in enumerate(drug_index_list) if value == drug]
    tissue_index_dict = {}
    for tissue in tissue_dict:
        tissue_list = []
        for index in drug_index:      # cell line id
            cell_name = cell_index_list[index]
            if csv_dict[cell_name] == tissue:
                tissue_list.append(index)
                tissue_index_dict[tissue] = tissue_list
    
    drug_tissue_cell_index[drug] = tissue_index_dict

with open('drug_cell_list/CCLE/' + 'all_' + 'drug_tissue_cell_index.json', 'w') as file:
    json.dump(drug_tissue_cell_index, file)