import csv
import json
import pandas as pd
# 用于存储构建的字典
csv_dict = {}

csv_file_path = 'global_data/PANCANCER_IC.csv'    # GDSC
# csv_file_path = '/workspace/geshuang/data/CCLE/secondary-screen-dose-response-curve-parameters.csv'
with open(csv_file_path, 'r', newline='') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)
    for row in csv_reader:
        csv_dict[row[3]] = row[5]       # GDSC: Cosmic sample Id:Tissue
with open('drug_cell_list/GDSC/cell_tissue.json', 'w') as file:
    json.dump(csv_dict, file)
########### Count the number of tissue types. ###############
df = pd.read_csv(csv_file_path, index_col=False)
df.dropna(subset=['Tissue'], inplace=True)
tissue_type = df['Tissue'].unique().tolist()
tissue_dict = {string: i for i, string in enumerate(tissue_type)}
with open('drug_cell_list/GDSC/tissue_type.json', 'w') as file:
    json.dump(tissue_dict, file)

############################# Save the indices of cells under the current drug treatment according to tissue type in the processed dataset .pt file. {'tissue_type':[cell_index1, cell_index2]}
# drug = 'Gefitinib'
# drug = 'Erlotinib'
# drug = 'AR-42'
# drug = 'Vorinostat'
# with open('drug_cell_list/tissue_type.json', 'r') as file:   # read all tissue types
#     tissue_dict = json.load(file)
# with open('drug_cell_list/cell_tissue.json', 'r') as file:   # read "cell names：tissue" dict
#     csv_dict = json.load(file)

# cell_index_list = []
# with open('drug_cell_list/' + drug + '_cell_list.txt', 'r') as f:      # 
#     for line in f:
#         cell_index_list.append(line.strip())

# tissue_index_dict = {}
# for tissue in tissue_dict:
#     tissue_list = []
#     for index, cell_name in enumerate(cell_index_list):
#         if csv_dict[cell_name] == tissue:
#             tissue_list.append(index)
#             tissue_index_dict[tissue] = tissue_list

# with open('drug_cell_list/' + drug + '_tissue_cell_index.json', 'w') as file:  
#     json.dump(tissue_index_dict, file)




