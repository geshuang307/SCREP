import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat_gcn_transformer_ge_only_pretrain_meta import FeatureRelationNetwork
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score
from utils_sc import *
import datetime
import argparse
import random
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics.cluster import adjusted_rand_score
import torch.nn.functional as F
import scipy as sp
from sklearn.model_selection import KFold
from scipy.stats import kendalltau, pearsonr
# import scipy.stats
loss_fn = nn.MSELoss()
# training function at each epoch
import json
class meta:
    def __init__(self,args):
        
        self.model = FeatureRelationNetwork(n_output=1)
        ####################### load bulk pretrained weights ######################
        if args.pretrained_path != None:
            self.model.load_state_dict(torch.load(args.pretrained_path), strict = False)
        # ################### load sc pretrained weights ######################
        if args.sc_pretrained_encoder != None:
            self.model.FeatureEncoder.load_state_dict(torch.load(args.sc_pretrained_encoder), strict = True)
            self.model.RelationNetwork.load_state_dict(torch.load(args.sc_pretrained_classify), strict = True)

        # self.number_of_metadata = args.example_number + args.query_size
        # self.example_number = args.example_number
        # self.query_size = args.query_size
        # self.relation_model = self.model.OneClassifier
        # self.relation_model = self.model.RelationNetwork
        self.update_step = 10
        self.update_lr = 0.01
        self.lr = args.lr
        self.early_stop1 = True
        self.cuda_name = args.cuda_name
        self.device = torch.device(self.cuda_name if torch.cuda.is_available() else "cpu")
        if args.datalist is not None:
            self.data_list = args.datalist
        elif args.dataset == 'GDSC_norm':
            with open('drug_cell_list/GDSC/all_drug_tissue_cell_index.json', 'r') as file:
                self.all_drug_tissue_cell_index = json.load(file)
                self.data_list = list(self.all_drug_tissue_cell_index.keys())
                self.train_data = TestbedDataset(root = 'data', dataset = 'GDSC_norm_all_continueic50_gene16906_index')
        elif args.dataset == 'CCLE':
            with open('drug_cell_list/CCLE/all_drug_tissue_cell_index.json', 'r') as file:
                self.all_drug_tissue_cell_index = json.load(file)
                self.data_list = list(self.all_drug_tissue_cell_index.keys())
                self.train_data = TestbedDataset(root = 'data', dataset = 'CCLE_maxmin_all_auc_gene16906_index')
            
            
        # self.encode_model = self.model.FeatureEncoder.to(self.device)
        # self.relation_model = self.relation_model.to(self.device)
        self.model = self.model.to(self.device)
        # self.optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, self.model.FeatureEncoder.parameters())}, \
        #     {'params': self.relation_model.parameters()}], lr=0.001)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        # Set learning rate scheduler 
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = args.step_size, gamma=0.1)  # 0.5
        
    def sample_test_split(self, data0_loader, data1_loader):
        for i, data0 in enumerate(data0_loader):
            data0 = data0.to(self.device)
            samples = data0
            sample_labels = data0.y
            break
        for i, data1 in enumerate(data1_loader):
            data1 = data1.to(self.device)
            querys = data1
            query_labels = data1.y
            break
        
        return samples, sample_labels, querys, query_labels

    # def meta_train(self, args, samples, samples_label, querys, query_labels):
        
    #     sample_features = torch.cat([self.model.FeatureEncoder(samples[index]) for index in range(len(samples))], dim =0)  # 10, 1000
    #     query_features = torch.cat([self.model.FeatureEncoder(querys[index]) for index in range(len(querys))], dim = 0)
    
    #     logits = self.relation_model(sample_features)
    #     loss = loss_fn(logits.squeeze(1), samples_label)
    #     grad = torch.autograd.grad(loss, self.relation_model.parameters())
    #     fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.relation_model.parameters())))
    #     logits_q = self.relation_model(query_features, fast_weights)    

    #     for _ in range(1, 100):
    #         logits = self.relation_model(sample_features, fast_weights)
    #         loss = loss_fn(logits.squeeze(1), samples_label)
    #         grad = torch.autograd.grad(loss, fast_weights)
    #         fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
    #         logits_q = self.relation_model(query_features, fast_weights)
    #     return logits_q

    def meta_train(self, args, samples, samples_label, querys, query_labels):
        
        sample_features = torch.cat([self.model.FeatureEncoder(samples[index]) for index in range(len(samples))], dim=0)  # 10, 1000
        query_features = torch.cat([self.model.FeatureEncoder(querys[index]) for index in range(len(querys))], dim=0)

        logits = self.model.RelationNetwork(sample_features)

        loss = loss_fn(logits.squeeze(1), samples_label)

        grad_encoder = torch.autograd.grad(loss, self.model.FeatureEncoder.parameters(), allow_unused=True, retain_graph=True)
        grad_relation_model = torch.autograd.grad(loss, self.model.RelationNetwork.parameters())

        fast_weights_encoder = list(map(lambda p: p[1] - self.update_lr * p[0] if p[0] is not None else p[1], zip(grad_encoder, self.model.FeatureEncoder.parameters())))
        fast_weights_relation_model = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_relation_model, self.model.RelationNetwork.parameters())))

        logits_q = self.model.RelationNetwork(query_features, fast_weights_relation_model)

        for _ in range(1, 100):

            logits = self.model.RelationNetwork(sample_features, fast_weights_relation_model)
            loss = loss_fn(logits.squeeze(1), samples_label)
            
            grad_encoder = torch.autograd.grad(loss, fast_weights_encoder, allow_unused=True, retain_graph=True)
            grad_relation_model = torch.autograd.grad(loss, fast_weights_relation_model)

            fast_weights_encoder = list(map(lambda p: p[1] - self.update_lr * p[0] if p[0] is not None else p[1], zip(grad_encoder, fast_weights_encoder)))
            fast_weights_relation_model = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad_relation_model, fast_weights_relation_model)))

            logits_q = self.model.RelationNetwork(query_features, fast_weights_relation_model)
        return logits_q
    
    def meta_test(self, args, samples, samples_label, querys, query_labels):
        sample_features = torch.cat([self.model.FeatureEncoder(samples[index]) for index in range(len(samples))], dim =0)  # 10, 1000
        query_features = torch.cat([self.model.FeatureEncoder(querys[index]) for index in range(len(querys))], dim = 0)
        # C=2, 5， 1000 
        logits = self.model.RelationNetwork(sample_features)
        samples_label = torch.stack(samples_label)
        loss = loss_fn(logits, samples_label)
        grad = torch.autograd.grad(loss, self.model.RelationNetwork.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.model.RelationNetwork.parameters())))
        logits_q = self.model.RelationNetwork(query_features, fast_weights)
        for _ in range(1, self.update_step):
            logits = self.model.RelationNetwork(sample_features, fast_weights)
            loss = loss_fn(logits, samples_label)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            logits_q = self.model.RelationNetwork(query_features, fast_weights)
        
        return logits_q
    
    def caculate_metrix(self, args):

        self.model.eval()
        result_file_name ='result/'+ args.dataset +'/result_' + args.model_st + '_' + args.dataset + args.exp_name + '_shot' + str(args.example_number) + '.csv'
        encode_model_file_name = 'encode_model_' + args.model_st + '_' + args.dataset + args.exp_name + '_shot' + str(args.example_number) +  '.model'
        relation_model_file_name = 'relation_model_' + args.model_st + '_' + args.dataset + args.exp_name + '_shot' + str(args.example_number) + '.model'
        test_batch = self.train_data
        test_loader = DataLoader(test_batch, batch_size=1, shuffle=False)
        label_list = []
        logits_list = []
        for i, data in enumerate(test_loader):
            data = data.to(self.device)
            test_label = data.y
            sample_features = self.model.FeatureEncoder(data)
            logits = self.model.RelationNetwork(sample_features)
            # logits = F.softmax(logits, dim=1)
            label_list.append(test_label.detach().cpu().numpy())
            logits_list.append(logits.squeeze(1).detach().cpu().numpy())
        torch.save(self.model.FeatureEncoder.state_dict(), 'result/'+ args.dataset + '/' +'last_' + encode_model_file_name)
        torch.save(self.model.RelationNetwork.state_dict(), 'result/'+ args.dataset + '/' +'last_' + relation_model_file_name)
        label_all = np.array(label_list)
        logits_all = np.array(logits_list)
        P = logits_all
        G = label_all
        result = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P)]
        print(result)

        return result
    
    def test_other(self):
        test0_batch = self.test0_data
        test1_batch = self.test1_data
        test_batch = test0_batch + test1_batch
        test_loader = DataLoader(test_batch, batch_size=1, shuffle=True)
        label_list = []
        logits_list = []
        for i, data in enumerate(test_loader):
            data = data.to(self.device)
            test_label = data.y
            sample_features = self.model.FeatureEncoder(data)
            logits = self.model.RelationNetwork(sample_features)
            logits = F.softmax(logits, dim=1)
            label_list.append(test_label.detach().cpu().numpy())
            logits_list.append(logits.detach().cpu().numpy())
        label_all = np.array(label_list)
        logits_all = np.array(logits_list).squeeze(1)
        sens_predict = logits_all[:, 1]
        AUC = roc_auc_score(label_all, sens_predict)
        AP = average_precision_score(label_all, sens_predict)
        predict_labels = np.argmax(logits_all, axis=1) 
        confusionmatrix = confusion_matrix(label_all, predict_labels)
        classification = classification_report(label_all, predict_labels, target_names=['Sensitive', 'Resistant'], digits=4)
        result = [confusionmatrix, classification, AUC, AP]
        print(confusionmatrix)
        print(classification)
        print(AUC)
        print(AP)


    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0*np.array(data)
        n = len(a)
        m, se = np.mean(a), sp.stats.sem(a)
        h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
        return m,h
    
    def test(self, args):

        self.model.eval()
        rmse_list = []
        mse_list = []
        person_list = []
        spearman_list = []
        kendalltau_list = []
        all_label_list = []
        all_logits_list, all_logit_list = [], []
        for drug in self.data_list:
            self.tissue_cell_index = self.all_drug_tissue_cell_index[drug]
            concatenate_list = [item for sublist in self.tissue_cell_index.values() for item in sublist]
            tissue_batch = self.train_data[concatenate_list]
            test_loader = DataLoader(tissue_batch, batch_size=1, shuffle=False)
            label_list = []
            logits_list = []
            for i, data in enumerate(test_loader):
                data = data.to(self.device)
                test_label = data.y
                sample_features = self.model.FeatureEncoder(data)
                logits = self.model.RelationNetwork(sample_features)
                label_list.append(test_label.detach().cpu().numpy())
                logits_list.append(logits.squeeze(1).detach().cpu().numpy())

            label_all = np.array(label_list)
            logits_all = np.array(logits_list)
            P = logits_all
            G = label_all
            rmse_list.append(rmse(G,P))
            mse_list.append(mse(G,P))
            person_list.append(pearsonr(G.squeeze(),P.squeeze()))
            kendalltau_list.append(kendalltau(G,P))
            spearman_list.append(spearman(G,P))
            all_label_list.extend(label_list)
            all_logit_list.extend(logits_list)
        result_list = [rmse_list, mse_list, kendalltau_list, spearman_list, person_list]
        np.save('result/plot_bulk/' + args.dataset + '_' + args.exp_name + 'result_list.npy', result_list)
        label_logit = [all_label_list, all_logit_list]
        np.save('result/plot_bulk/' + args.dataset + '_' + args.exp_name + 'label_logit_list.npy', label_logit)
        return 

        # Set distribution loss 

    # loss_disrtibution = loss
    def main(self, args):

        self.model.train()
        best_loss = 1000
        query_acc_list, loss_list, mean_acc = [], [], []
        encode_model_file_name = 'encode_model_' + args.model_st + '_' + args.dataset + args.exp_name + '_shot' + str(args.example_number) +  '.model'
        relation_model_file_name = 'relation_model_' + args.model_st + '_' + args.dataset + args.exp_name + '_shot' + str(args.example_number) + '.model'
        for iter in range(args.num_iters+1):         
            ########################## all 223 drugs #######################################
            
            dataset_choice = random.choice(self.data_list)     # randomly choose one drug 
            self.tissue_cell_index = self.all_drug_tissue_cell_index[dataset_choice]   # all tissues_cell indexes related to the drug
            random_tissues = random.sample(self.tissue_cell_index.keys(), 12)  # randomly choose 8 tissues
            # Concatenate the lists corresponding to these keys
            random_tissues1, random_tissues2 = [], []
            for key in random_tissues[:int(len(random_tissues)/2)]:
                random_tissues1.extend(self.tissue_cell_index[key])
            for key in random_tissues[int(len(random_tissues)/2):]:
                random_tissues2.extend(self.tissue_cell_index[key])
            tissue_batch1 = self.train_data[random_tissues1]
            tissue_batch2 = self.train_data[random_tissues2]
            len_min = min(len(tissue_batch1), len(tissue_batch2))
            data0_loader = DataLoader(tissue_batch1, batch_size=len_min, shuffle=True)
            data1_loader = DataLoader(tissue_batch2, batch_size=len_min, shuffle=True)
            samples, samples_label, querys, query_labels = self.sample_test_split(data0_loader, data1_loader)
            self.optimizer.zero_grad()
            logits_q = self.meta_train(args, samples, samples_label, querys, query_labels)

            loss_mse = loss_fn(logits_q.squeeze(1), query_labels)

            G = query_labels.detach().cpu()
            P = logits_q.squeeze().detach().cpu()
            ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P)]
            loss_mse.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            loss_list.append(loss_mse)
            # query_acc_list.append(query_acc)
            # plot_acc_list.append(query_acc)
            print('num_of_sample {}, iter {}, loss {}, accuracy of meta training on sc is {}'.format(len_min, iter, loss_mse, ret))
            print('encoder learning_rate {}'.format(self.optimizer.param_groups[0]['lr']))
           
            if (iter + 1) % 100 == 0 and iter+1 >= 200:
                print("train episode:",iter+1," loss:", loss_mse.data, " average_acc:", np.mean(query_acc_list))
                print('######################################################################################')
                print('######################################################################################')
                mean_acc.append(np.mean(query_acc_list))
                # self.caculate_metrix(args)
                if self.early_stop1 and (np.mean(query_acc_list) >= 0.999):
                    print("episode:",iter + 1, "early stop")
                    break
                query_acc_list = []
            if loss_mse < best_loss and iter > 7000:
                best_loss = loss_mse
                torch.save(self.model.FeatureEncoder.state_dict(), 'result/'+ args.dataset + '/' + 'best_loss_' + encode_model_file_name)
                torch.save(self.model.RelationNetwork.state_dict(), 'result/'+ args.dataset + '/' + 'best_loss_'+ relation_model_file_name)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train model')

    # parser.add_argument('--dataset', type = str, default = 'GDSC_norm')
    parser.add_argument('--dataset', type = str, default = 'CCLE')
    parser.add_argument('--datalist', type = str, default = None)
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    # parser.add_argument('--num_iters', type=int, required=False, default=22400, help='Number of epoch')   # GDSC
    parser.add_argument('--num_iters', type=int, required=False, default=14340, help='Number of epoch')     # CCLE
    parser.add_argument('--test_iters', type = int, default = 600)
    parser.add_argument('--log_interval', type=int, required=False, default=200, help='Log interval')
    parser.add_argument('--cuda_name', type=str, required=False, default="cuda:4", help='Cuda')
    parser.add_argument('--class_number', type = int, default = 2)
    parser.add_argument('--example_number', type = int, default = 0)    
    parser.add_argument('--query_size', type = int, default=0)
    parser.add_argument('--feature_dim', type = int, default = 256)
    parser.add_argument('--step_size', type = int, default = 7000)         # 200
    # parser.add_argument('--pretrained_path', type = str, default = 'model_FeatureRelationNetwork_GDSC_norm_continue_ic50.model')    # GDSC_norm continuous value
    parser.add_argument('--pretrained_path', type = str, default = 'model_FeatureRelationNetwork_CCLE_norm_continue_auc.model')     # CCLE_norm continue 
    parser.add_argument('--sc_pretrained_encoder', type = str, default = None)
    parser.add_argument('--exp_name', type = str, default = 'all_drug')
    parser.add_argument('--model_st', type = str, default = 'FeatureRelationNetwork_meta_tissue20')
    parser.add_argument('--save_at', type = int, default = 500)    
    args = parser.parse_args()

    # seeds = [1934,1944,1954,1964, 1974,1984, 1994,2004, 2014,2024]
    seeds = [2024]
    plot_acc_, AUC_list, AP_list = [], [], []
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for seed in seeds:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        META = meta(args)
        # META.test(args)
        META.main(args)
        result = META.caculate_metrix(args)
        [confusionmatrix, classification, AUC, AP] = result
        AUC_list.append(AUC)
        AP_list.append(AP)
    # np.save('result/plot_AUC/' + args.dataset + '_' +args.exp_name + '_shot' + str(args.example_number) + '.npy', AUC_list)
    # np.save('result/plot_AP/' + args.dataset + '_' +args.exp_name + '_shot' + str(args.example_number) + '.npy', AP_list)