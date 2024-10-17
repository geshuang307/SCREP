import pandas as pd
import requests
import time

################ Download 24 drugs smiles from CCLE ##################
# df = pd.read_csv("/workspace/geshuang/data/CCLE/CCLE_NP24.2009_Drug_data_2015.02.24.csv")
# df = pd.read_csv("/workspace/geshuang/data/CCLE/secondary-screen-dose-response-curve-parameters.csv")    # PRISM Repurposing 19Q4 Files
df = pd.read_excel("/workspace/geshuang/data/GES157220_SCP542/41588_2020_726_MOESM3_ESM-s1-s10.xlsx", sheet_name='Table S9', header=2, skiprows=1, index_col=0)
# df = df.iloc[3:]
drug = df['Compound'].tolist()
drug = [drug[index].split('(')[0] for index in range(len(drug))]
drug = list(set(drug))
drug = pd.DataFrame(drug, columns=['name'])
#############
#######
#### get SMILES
#######

def replace_url(url_new1, element):
    # replace {} to element
    return url_new1.format(element)

################ query CID and save  ###############
url_cid= "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/cids/txt"
new_cid_urls = drug['name'].apply(lambda x: replace_url(url_cid, x))        # list 
cids = []
for url in new_cid_urls:
    try:
        response = requests.get(url, verify=False)
        time.sleep(1)
        cids.append(response.text)
        print(response.text)
    except Exception as e:
        print("Error", str(e))
        cids.append("")

######### put cid into drug dict ###############
drug['CID'] = cids
############### query CanonicalSMILES and save ###############
url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES/txt"

# use apply to replace every element
new_urls = drug['name'].apply(lambda x: replace_url(url, x))        # list 

# save results 
smi = []

# Traverse a new list of URLs and make requests or perform other actions
for url in new_urls:
    try:
        response = requests.get(url, verify=False)
        time.sleep(1)
        smi.append(response.text)
        print(response.text)
    except Exception as e:
        print("Error", str(e))
        smi.append("")

#########put smi into drug dict
drug['CanonicalSMILES'] = smi

################ query IsomericSMILES and save ###############
url_smiles2 = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/IsomericSMILES/txt"
new_smiles2_urls = drug['name'].apply(lambda x: replace_url(url_smiles2, x))        # list 
IsomericSMILES = []
for url in new_smiles2_urls:
    try:
        response = requests.get(url, verify=False)
        time.sleep(1)
        IsomericSMILES.append(response.text)
        print(response.text)
    except Exception as e:
        print("Error", str(e))
        IsomericSMILES.append("")
drug['IsomericSMILES'] = IsomericSMILES
#########save
# drug.to_excel("drug_smiles_ccle1448_own.xlsx",index=False)
drug.to_excel("test/drug_smiles_GSE157220_own.xlsx",index=False)


