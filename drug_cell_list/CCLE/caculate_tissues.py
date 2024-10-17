import json

# with open('drug_cell_list/CCLE/all_drug_tissue_cell_index.json', 'r') as file:   # read all tissue types
#     drug_dict = json.load(file)

# lenth_list = []
# for key in drug_dict.keys():
#     lenth = len(drug_dict[key])
#     lenth_list.append(lenth)
# print('min:', min(lenth_list))
# print('max:', max(lenth_list))

with open('drug_cell_list/GDSC/all_drug_tissue_cell_index.json', 'r') as file:   # read all tissue types
    drug_dict = json.load(file)

lenth_list = []
for key in drug_dict.keys():
    lenth = len(drug_dict[key])
    lenth_list.append(lenth)
print('min:', min(lenth_list))
print('max:', max(lenth_list))