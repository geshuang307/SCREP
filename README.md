# SCREP
The code will be released here later.
### Datasets
The published Panglao dataset was downloaded from https://panglaodb.se/, please put the downloaded data under the path: data/panglao_10000.h5ad

### Download single cell data from scDEAL[Nat Comm2022-Deep transfer learning of cancer drug responses by integrating bulk and single-cell RNA-seq data] and SCAD[Advanced science2023-Enabling Single‐Cell Drug Response Annotations from Bulk RNA‐Seq Using SCAD]:
GEO access     |Drug     |Cell line|Cancer type             |No. Res  |No. Sens| N.A.| Author|
GSE117872_HN120|Cisplatin|OSCC     |Oral squamous cell carcinomas|172 |346 |/ |Sharma, et al.|                                       
GSE117872_HN137|Cisplatin|OSCC     |Oral squamous cell carcinomas|150 |388 |/ |Sharma, et al.|
GSE149383      |Erlotinib|PC9      |Lung cancer                  |617 |849 |/ |Aissa, et al. |
GSE110894      |I-BET-762|leukaemic cells|Acute myeloidleukemia  |670 |719 |/ |Bell, et al.  |
GSE157220      |Gefitinib,Vorinostat,AR-42|SCC47|Head and Neck Cancer|162|162|472|Kinker et.al|
GSE157220      |NVP-TAE684,Afatinib,Sorafenib|JHU006|Head and Neck Cancer|81|81|259|Kinker et.al|
GSE228154      |Afatinib |MDA-MB-468 cells|metastatic breast cancer|665|846|/ |J. M. McFarland|

### preprocessing GDSC data : preprocess_continue_IC50_ge16906_everydrug.py, you will get data files in "data/processed": 
          data/processed/GDSC_norm_all_continueic50_gene16906_index.pt

### preprocessing GDSC data into "sensitive" or "resistant": It is necessary to align the bulk data separately with the sensitive and resistant categories of the sc data. The IC50 should be filtered according to a 5% z-score, resulting in the following bulk data (sensitive: class 1, resistant: class 0):
          data/processed/GDSC_norm_allclass0_z_score_ic50_gene16906.pt
          data/processed/GDSC_norm_allclass1_z_score_ic50_gene16906.pt

### get indices of cell lines of specific tissue: run drug_cell_list/tissue.py and then run drug_cell_list/drug_tissue_cell_index.py:
          drug_cell_list/cell_tissue.json
          drug_cell_list/tissue_type.json
          drug_cell_list/all_drug_tissue_cell_index.json


### pretrain bulk model:
training_IC50_ge16906_fewshot_metatraining_2loss_onedrug.py

- Based on the general GDSC pretrained model: "model_FeatureRelationNetwork_GDSC_norm_continue_ic50.model", it is the model trained on all drug-cell lines dataset.

### pretrain sc model:
training_IC50_ge16906_fewshot_metatraining_2loss_mmd.py
- change parameters: "--dataset"，"--pretrained_path"，"--exp_name" to train sc model on different sc datasets.

*Thanks to [Thang Chu, et al.] for providing excellent code and documentation. This project was inspired by and includes some code from [GraTransDRP] T. Chu, T. T. Nguyen, B. D. Hai, Q. H. Nguyen and T. Nguyen, "Graph Transformer for Drug Response Prediction," in IEEE/ACM Transactions on Computational Biology and Bioinformatics, vol. 20, no. 2, pp. 1065-1072, 1 March-April 2023
*Thanks to [Gabriela S Kinker, et al] for providing partial datasets [Gabriela S Kinker, et al. Pan-cancer
single-cell rna-seq identifies recurring programs of cellular heterogeneity. Nature genetics, 52(11):1208–1218, 2020]
*Thanks to [Junyi Chen, et al.] for providing partial datasets[Chen, J., Wang, X., Ma, A., Wang, Q.E., Liu, B., Li, L., Xu, D. and Ma, Q., 2022. Deep transfer learning of cancer drug responses by integrating bulk and single-cell RNA-seq data. Nature Communications, 13(1), p.6494.]

### Tips:
decreasing the args.updata_lr when the accuracy of meta training is retaining around 0.5 and dose not raise during the training phase of single cell transferring. 
