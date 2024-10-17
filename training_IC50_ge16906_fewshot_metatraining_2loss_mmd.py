import numpy as np

import torch
import torch.nn as nn

# from models.gat_gcn_transformer_ge_only import GAT_GCN_Transformer_ge_only
from models.gat_gcn_transformer_ge_only_pretrain_meta import FeatureRelationNetwork
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score
from utils import *
import datetime
import argparse
import random
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics.cluster import adjusted_rand_score
import torch.nn.functional as F
import scipy as sp
import loss.mmd as mmd
from sklearn.model_selection import KFold
# import scipy.stats
loss_fn = nn.CrossEntropyLoss()

class meta:
    def __init__(self,args):
        
       
    
        self.model = FeatureRelationNetwork(n_output=1)
         ####################### load the pretrained weight of the bulk ######################
        if args.pretrained_path != None:
            self.model.load_state_dict(torch.load(args.pretrained_path), strict = False)
            a = torch.load(args.pretrained_path)
            if hasattr(self.model.FeatureEncoder.conv1, 'lin_src') and a['FeatureEncoder.conv1.lin_src.weight'] is not None:
                self.model.FeatureEncoder.conv1.lin.weight.data = a['FeatureEncoder.conv1.lin_src.weight'].data.clone()
                print(self.model.FeatureEncoder.conv1.lin.weight)
        # ################### load the pretrained sc model ######################
        if args.meta_pretrained_encoder != None:
            self.model.FeatureEncoder.load_state_dict(torch.load(args.meta_pretrained_encoder), strict = True)
            self.model.RelationNetwork.load_state_dict(torch.load(args.meta_pretrained_classify), strict = True)

        self.number_of_metadata = args.example_number + args.query_size
        self.example_number = args.example_number
        self.query_size = args.query_size
        # self.relation_model = self.model.OneClassifier
        self.relation_model = self.model.RelationNetwork
        #################### only for continous label ###############
        in_features = self.relation_model.fc2.out_features
        out = nn.Linear(in_features, out_features=2)
        self.relation_model.out = out


        self.update_step = 10
        self.update_lr = 0.01
        self.lr = args.lr
        self.early_stop1 = True
        # self.log_interval = args.log_interval
        self.cuda_name = args.cuda_name
        self.device = torch.device(self.cuda_name if torch.cuda.is_available() else "cpu")
        # self.train0_data = TestbedDataset(root = 'data', dataset = args.dataset + '_allclass0_z_score_auc_gene16906')        # CCLE
        # self.train1_data = TestbedDataset(root = 'data', dataset = args.dataset + '_allclass1_z_score_auc_gene16906') 
        self.train0_data = TestbedDataset(root = 'data', dataset = args.dataset + '_sc_allclass0_drug_gene16906')   # 单细胞
        self.train1_data = TestbedDataset(root = 'data', dataset = args.dataset + '_sc_allclass1_drug_gene16906')

        self.model = self.model.to(self.device)
        # self.optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, self.model.FeatureEncoder.parameters())}, \
        #     {'params': self.model.RelationNetwork.parameters()}], lr=0.0001)
        self.optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, self.model.FeatureEncoder.parameters())}, \
            {'params': self.relation_model.parameters()}], lr=0.0001)
        # Set learning rate scheduler 
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = args.step_size, gamma=0.5)  
        # self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.meta0_data = TestbedDataset(root = 'data', dataset = args.dataset + '_sc_allclass0_drug_gene16906')
        self.meta1_data = TestbedDataset(root = 'data', dataset = args.dataset + '_sc_allclass1_drug_gene16906')
        if 'GDSC' in args.exp_name:
            self.bulk0_data = TestbedDataset(root = 'data', dataset = 'GDSC_norm_allclass0_z_score_ic50_gene16906')
            self.bulk1_data = TestbedDataset(root = 'data', dataset = 'GDSC_norm_allclass1_z_score_ic50_gene16906')
            print('GDSC is the pretrain dataset')
        elif 'CCLE' in args.exp_name:
            self.bulk0_data = TestbedDataset(root = 'data', dataset = 'CCLE_norm_allclass0_z_score_auc_gene16906')
            self.bulk1_data = TestbedDataset(root = 'data', dataset = 'CCLE_norm_allclass1_z_score_auc_gene16906')
            print('CCLE is the pretrain dataset')
        else:
            pass
        if args.test_dataset is not None:
            self.test0_data =  TestbedDataset(root = 'data', dataset = args.test_dataset + '_sc_allclass0_drug_gene16906')
            self.test1_data =  TestbedDataset(root = 'data', dataset = args.test_dataset + '_sc_allclass1_drug_gene16906')
    def sample_test_split(self, data0_loader, data1_loader):
        for i, data0 in enumerate(data0_loader):
            data0 = data0.to(self.device)
            samples0 = data0[:self.example_number]
            sample_labels0 = data0.y[:self.example_number]
            querys0 = data0[self.example_number:]
            query_labels0 = data0.y[self.example_number:]
        for i, data1 in enumerate(data1_loader):
            data1 = data1.to(self.device)
            samples1 = data1[:self.example_number]
            sample_labels1 = data1.y[:self.example_number]
            querys1 = data1[self.example_number:]
            query_labels1 = data1.y[self.example_number:]
        samples = samples0 + samples1
        querys = querys0 + querys1
        sample_labels = list(sample_labels0) + list(sample_labels1)
        query_labels = list(query_labels0) + list(query_labels1)
        
        return samples, sample_labels, querys, query_labels

    def meta_train(self, args, samples, samples_label, querys, query_labels):
        
        sample_features = torch.cat([self.model.FeatureEncoder(samples[index]) for index in range(len(samples))], dim =0)  # 10, 1000
        query_features = torch.cat([self.model.FeatureEncoder(querys[index]) for index in range(len(querys))], dim = 0)
        # sample_features = sample_features.view(args.class_number, args.example_number, args.feature_dim)  # C=2, 5， 1000 
        logits = self.relation_model(sample_features)
        samples_label = torch.stack(samples_label).long()
        loss = loss_fn(logits, samples_label)
        grad = torch.autograd.grad(loss, self.relation_model.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.relation_model.parameters())))
        logits_q = self.relation_model(query_features, fast_weights)
        # logits_q = self.relation_model(query_features)
        

        for _ in range(1, 100):
            logits = self.relation_model(sample_features, fast_weights)
            loss = F.cross_entropy(logits, samples_label)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            logits_q = self.relation_model(query_features, fast_weights)
        predict_labels = F.softmax(logits_q, dim=1).argmax(dim=1)
        rewards = [1 if predict_labels[j]==query_labels[j] else 0 for j in range(len(predict_labels))]
        indices = [index for index, value in enumerate(rewards) if value == 0]

        encoder_features = torch.cat([sample_features, query_features], dim = 0)

        return logits_q, indices, encoder_features

    
    def meta_test(self, args, samples, samples_label, querys, query_labels):
        sample_features = torch.cat([self.model.FeatureEncoder(samples[index]) for index in range(len(samples))], dim =0)  # 10, 1000
        query_features = torch.cat([self.model.FeatureEncoder(querys[index]) for index in range(len(querys))], dim = 0)
        # sample_features = sample_features.view(args.class_number, args.example_number, args.feature_dim)  # C=2, 5， 1000 
        logits = self.model.RelationNetwork(sample_features)
        samples_label = torch.stack(samples_label).long()
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
    
    def caculate_metrix(self, args, filtered_list_0, filtered_list_1):
        # self.encode_model.eval()
        # self.relation_model.eval()
        self.model.eval()
        result_file_name ='result/'+ args.dataset +'/result_' + args.model_st + '_' + args.dataset + args.exp_name + '_shot' + str(args.example_number) + '.csv'
        encode_model_file_name = 'encode_model_' + args.model_st + '_' + args.dataset + args.exp_name + '_shot' + str(args.example_number) +  '.model'
        relation_model_file_name = 'relation_model_' + args.model_st + '_' + args.dataset + args.exp_name + '_shot' + str(args.example_number) + '.model'
        train0_batch = [self.meta0_data[index0] for index0 in filtered_list_0]
        train1_batch = [self.meta1_data[index1] for index1 in filtered_list_1]
        # train0_batch = self.meta0_data[self.number_of_metadata:]
        # train1_batch = self.meta1_data[self.number_of_metadata:]
        test_batch = train0_batch + train1_batch
        test_loader = DataLoader(test_batch, batch_size=1, shuffle=False)
        label_list = []
        logits_list = []
        for i, data in enumerate(test_loader):
            data = data.to(self.device)
            test_label = data.y.long()
            sample_features = self.model.FeatureEncoder(data)
            logits = self.relation_model(sample_features)
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
        if args.save is not None:
            with open(result_file_name,'w') as f:
                f.write(','.join(map(str, result)))
            torch.save(self.model.FeatureEncoder.state_dict(), 'result/'+ args.dataset + '/' + encode_model_file_name)
            torch.save(self.relation_model.state_dict(), 'result/'+ args.dataset + '/' + relation_model_file_name)
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
            test_label = data.y.long()
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
        query_acc_list, test_aris = [], []

        self.model.eval()
        for iter in range(args.test_iters + 1):
            sampled_indices = random.sample(range(self.number_of_metadata, len(self.meta0_data)), self.number_of_metadata)
            train0_batch = [self.meta0_data[i] for i in sampled_indices]
            sampled_indices = random.sample(range(self.number_of_metadata, len(self.meta1_data)), self.number_of_metadata)
            train1_batch = [self.meta1_data[i] for i in sampled_indices]
            meta0_loader = DataLoader(train0_batch, batch_size=self.number_of_metadata, shuffle=True)
            meta1_loader = DataLoader(train1_batch, batch_size=self.number_of_metadata, shuffle=True)
            samples, samples_label, querys, query_labels = self.sample_test_split(meta0_loader, meta1_loader)
            logits_q = self.meta_test(args, samples, samples_label, querys, query_labels)
            query_labels = torch.stack(query_labels).long()
            # loss_cross = F.cross_entropy(logits_q, query_labels)
            predict_labels = F.softmax(logits_q, dim=1).argmax(dim=1)
            rewards = [1 if predict_labels[j]==query_labels[j] else 0 for j in range(len(predict_labels))]
            query_acc = np.sum(rewards) / len(query_labels)
                ############ 保留一下128维的药物向量 ############
            query_acc_list.append(query_acc)
       
            ari = adjusted_rand_score(predict_labels.detach().cpu().data, query_labels.detach().cpu().data)
            test_aris.append(ari)
            print('test iter {}, accuracy of meta test on sc is {}'.format(iter, query_acc))
            print('model learning_rate of sc meta test: {}'.format(self.optimizer.param_groups[0]['lr']))
            if (iter + 1) % 101 == 0:
                print('iter {}, mean accuracy on sc is {}'.format(iter + 1, np.mean(query_acc_list)))
                print('##########################################################################')
                print('##########################################################################')
                if self.early_stop1 and (np.mean(query_acc_list) > 0.90):
                    print("test episode: ", str(iter + 1) , "early stop")
                    break
                query_acc_list = []

        
        return 

        # Set distribution loss 
    def dist_loss(self, x,y, GAMMA= 1000):
        result = mmd.mmd_loss(x,y,GAMMA)
        return result

    # loss_disrtibution = loss
    def main(self, args):

        self.model.train()
        best_loss = 1000
        query_acc_list, loss_list, mean_acc, plot_acc_list = [], [], [], []
        encode_model_file_name = 'encode_model_' + args.model_st + '_' + args.dataset + args.exp_name + '_shot' + str(args.example_number) +  '.model'
        relation_model_file_name = 'relation_model_' + args.model_st + '_' + args.dataset + args.exp_name + '_shot' + str(args.example_number) + '.model'
        result_file_name = 'result_' + args.model_st + '_' + args.dataset + args.exp_name + '_shot' + str(args.example_number) + '.csv'
        loss_fig_name = 'model_' + args.model_st + '_' + args.dataset + args.exp_name + '_shot' + str(args.example_number) +'_loss'
        acc_fig_name = 'model_' + args.model_st + '_' + args.dataset + args.exp_name + '_shot' + str(args.example_number) + '_acc'
        exclude_indices_0 = random.sample(range(len(self.train0_data)), self.number_of_metadata)
        filtered_list_0 = [index for index in range(len(self.train0_data)) if index not in exclude_indices_0]
        exclude_indices_1 = random.sample(range(len(self.train1_data)), self.number_of_metadata)
        filtered_list_1 = [index for index in range(len(self.train1_data)) if index not in exclude_indices_1]
        plot_acc = []
        for iter in range(args.num_iters+1):
            
            train1_batch = [self.train1_data[i] for i in exclude_indices_1]        
            train0_batch = [self.train0_data[i] for i in exclude_indices_0]
            ############### bulk数据随机选 ######################
            bulk0_index = random.sample(range(len(self.bulk0_data)), self.number_of_metadata)
            bulk0_data = [self.bulk0_data[i] for i in bulk0_index]
            bulk1_index = random.sample(range(len(self.bulk1_data)), self.number_of_metadata)
            bulk1_data = [self.bulk1_data[i] for i in bulk1_index]
            ############### bulk数据固定选 ######################
            bulk0_loader = DataLoader(bulk0_data, batch_size=self.number_of_metadata, shuffle=True)   # 15的倍数
            bulk1_loader = DataLoader(bulk1_data, batch_size=self.number_of_metadata, shuffle=True)
            ####################################################
            samples_bulk, samples_label_bulk, querys_bulk, query_labels_bulk = self.sample_test_split(bulk0_loader, bulk1_loader)

            data0_loader = DataLoader(train0_batch, batch_size=self.number_of_metadata, shuffle=True)
            data1_loader = DataLoader(train1_batch, batch_size=self.number_of_metadata, shuffle=True)
            samples, samples_label, querys, query_labels = self.sample_test_split(data0_loader, data1_loader)
            
            self.optimizer.zero_grad()
            logits_q, _, encoder_features = self.meta_train(args, samples, samples_label, querys, query_labels)
            logits_q_bulk, _, bulk_encoder_features = self.meta_train(args, samples_bulk, samples_label_bulk, querys_bulk, query_labels_bulk)
            # Calculate meta-train loss
            query_labels = torch.stack(query_labels).long()
            loss_cross = loss_fn(logits_q, query_labels)
            # Calculate meta-train accuracy
            predict_labels = F.softmax(logits_q, dim=1).argmax(dim=1)
            rewards = [1 if predict_labels[j]==query_labels[j] else 0 for j in range(len(predict_labels))]

            
            query_acc = np.sum(rewards) / len(query_labels)
            if 'nopretrain' in args.exp_name:
                loss_all = loss_cross
            elif 'onlyalian' in args.exp_name:
                distribute_loss = self.dist_loss(encoder_features, bulk_encoder_features)
                loss_all = distribute_loss
            else:
                distribute_loss = self.dist_loss(encoder_features, bulk_encoder_features)
                loss_all = loss_cross + distribute_loss                   
            loss_all.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            loss_list.append(loss_all)
            query_acc_list.append(query_acc)
            plot_acc_list.append(query_acc)
            print('iter {}, loss {}, accuracy of meta training on sc is {}'.format(iter, loss_cross, query_acc))
            print('encoder learning_rate {}'.format(self.optimizer.param_groups[0]['lr']))
            # ret = [confusion_matrix(G, P), classification_report(G, P, target_names=['Sensitive', 'Resistant'], digits=4), AUC, AP]
            if (iter + 1) % 100 == 0 and iter+1 >= 100:
                print("train episode:",iter+1," loss:", loss_cross.data, " average_acc:", np.mean(query_acc_list))
                print('########################################################################################')
                print('########################################################################################')
                mean_acc.append(np.mean(query_acc_list))
                
                if self.early_stop1 and (np.mean(query_acc_list) >= 0.999):
                    print("episode:",iter + 1, "early stop")

                    break
                query_acc_list = []
            if loss_cross < best_loss and iter % args.save_at == 0 and iter > 0:
                best_loss = loss_cross
            if (iter + 1) % 20 == 0:
                plot_acc.append(np.mean(plot_acc_list))   # 50次迭代的准确度取平均
                plot_acc_list = []
                result = self.caculate_metrix(args, filtered_list_0, filtered_list_1)
                print(result)
        result = self.caculate_metrix(args, filtered_list_0, filtered_list_1)
        
        return plot_acc, result



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train model')
    # parser.add_argument('--model', type=int, required=False, default=4, help='0: Transformer_ge_mut_meth, 1: Transformer_ge_mut, 2: Transformer_meth_mut, 3: Transformer_meth_ge, 4: Transformer_ge, 5: Transformer_mut, 6: Transformer_meth')
    # parser.add_argument('--model', type=int, required=False, default=0)
    # parser.add_argument('--dataset', type = str, default = 'GSE149383')
    # parser.add_argument('--dataset', type = str, default = 'GSE140440')
    # parser.add_argument('--dataset', type = str, default = 'GSE110894')
    # parser.add_argument('--dataset', type = str, default = 'GSE112274')
    parser.add_argument('--dataset', type = str, default = 'GSE157220_JHU006')
    # parser.add_argument('--dataset', type = str, default = 'GSE157220_SCC47')
    # parser.add_argument('--dataset', type = str, default = 'MDAMB468')
    # parser.add_argument('--dataset', type = str, default = 'GSE117872_HN120')
    # parser.add_argument('--dataset', type = str, default = 'GSE117872_HN137')
    # parser.add_argument('--dataset', type = str, default = 'CCLE_norm')
    # parser.add_argument('--dataset', type = str, default = 'GSE157220_ALL')
    # parser.add_argument('--dataset', type = str, default = 'GSE149215')
    parser.add_argument('--test_dataset', type = str, default = None)

    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--num_iters', type=int, required=False, default=1000, help='Number of epoch')
    parser.add_argument('--test_iters', type = int, default = 600)
    parser.add_argument('--log_interval', type=int, required=False, default=200, help='Log interval')
    parser.add_argument('--cuda_name', type=str, required=False, default="cuda:5", help='Cuda')
    parser.add_argument('--class_number', type = int, default = 2)
    parser.add_argument('--example_number', type = int, default = 5)      # 样本量越大，泛化能力更强，相应的训练时间也会变长,有的数据集更多的样本会带来更大的准确性，有的数据集不需要
    parser.add_argument('--query_size', type = int, default=10)
    parser.add_argument('--feature_dim', type = int, default = 256)
    # parser.add_argument('--lrG', type = float, default = 0.5)
    # parser.add_argument('--lrS', type = int, default = 10000)
    parser.add_argument('--step_size', type = int, default = 200)         # 200
    # parser.add_argument('--pretrained_path', type = str, default = 'model_GAT_GCN_Transformer_ge_only_GDSC_z_score_noalin.model')
    # parser.add_argument('--pretrained_path', type = str, default = 'model_GAT_GCN_Transformer_ge_only_GDSC_z_score_metapretrained.model')  # GDSC
    # parser.add_argument('--pretrained_path', type = str, default = 'model_GAT_GCN_Transformer_ge_only_GDSC_norm_z_score_metapretrained.model')  # GDSC_norm 离散
    # parser.add_argument('--pretrained_path', type = str, default = 'model_GAT_GCN_Transformer_ge_only_CCLE_z_score_metapretrained.model')    # CCLE 离散
    # parser.add_argument('--pretrained_path', type = str, default = 'model_GAT_GCN_Transformer_ge_only_CCLE_norm_z_score_metapretrained.model') # CCLE_norm 离散
    # parser.add_argument('--pretrained_path', type = str, default = 'model_GAT_GCN_Transformer_ge_only_CCLE_GDSC_z_score_metapretrained.model')
    # parser.add_argument('--pretrained_path', type = str, default = 'model_gat_gcn_transformer_ge_only_pretrain_meta_GSE149383_z_score_metapretrained_sc_.model')
    # parser.add_argument('--pretrained_path', type = str, default = 'model_FeatureRelationNetwork_GDSC_norm_continue_ic50.model')    # GDSC_norm 连续
    # parser.add_argument('--pretrained_path', type = str, default = 'model_FeatureRelationNetwork_CCLE_norm_continue_auc.model')     # CCLE_norm 连续
    parser.add_argument('--pretrained_path', type = str, default = None)                      
    # parser.add_argument('--sc_pretrained_encoder', type = str, default = 'result/GSE149383/encode_model_FeatureRelationNetwork_meta_GSE149383GDSC_norm_continue_ic50_shot5.model')
    # parser.add_argument('--sc_pretrained_classify', type = str, default = 'result/GSE149383/relation_model_FeatureRelationNetwork_meta_GSE149383GDSC_norm_continue_ic50_shot5.model')
    # parser.add_argument('--sc_pretrained_encoder', type = str, default = None)
    parser.add_argument('--meta_pretrained_encoder', type = str, default = None)
    parser.add_argument('--meta_pretrained_encoder', type = str, default = 'result/CCLE/last_encode_model_FeatureRelationNetwork_meta_tissue20_CCLEall_drug_shot0.model')
    parser.add_argument('--meta_pretrained_classify', type = str, default = 'result/CCLE/last_relation_model_FeatureRelationNetwork_meta_tissue20_CCLEall_drug_shot0.model')
    # parser.add_argument('--meta_pretrained_encoder', type = str, default = 'result/GDSC_norm/encode_model_FeatureRelationNetwork_meta_GDSC_normErlotinib_only_shot0.model')
    # parser.add_argument('--meta_pretrained_classify', type = str, default = 'result/GDSC_norm/relation_model_FeatureRelationNetwork_meta_GDSC_normErlotinib_only_shot0.model')
    parser.add_argument('--save', type = str, default = None)
    # parser.add_argument('--exp_name', type = str, default = 'GDSC_norm_continue_ic50_mmd_meta_bulkshuffle')
    # parser.add_argument('--exp_name', type = str, default = 'GDSC_norm_continue_ic50_mmd')
    # parser.add_argument('--exp_name', type = str, default = 'GDSC_norm_continue_ic50_mmd_bulkshuffle')
    # parser.add_argument('--exp_name', type = str, default = 'GDSC_norm_continue_ic50_mmd_bulkshuffle_freezeencoder')
    parser.add_argument('--exp_name', type = str, default = 'CCLE_norm_continue_auc_mmd')
    # parser.add_argument('--exp_name' , type = str, default = 'GDSC_norm_continue_ic50_mmd_nopretrain')
    # parser.add_argument('--exp_name', type = str, default = 'zscore_metatraining_nopretrain')
    # parser.add_argument('--exp_name', type = str, default = 'GDSC_norm_zscore_mmd_bulkshuffle')
    # parser.add_argument('--exp_name', type = str, default = 'zscore_metatraining_CCLE')
    # parser.add_argument('--exp_name', type = str, default = 'zscore_metatraining_CCLE_norm')
    # parser.add_argument('--exp_name', type = str, default = 'zscore_metatraining_CCLE_GDSC')
    # parser.add_argument('--exp_name', type = str, default = 'CCLE_norm_continue_auc')
    parser.add_argument('--model_st', type = str, default = 'FeatureRelationNetwork_meta')
    parser.add_argument('--save_at', type = int, default = 500)    
    args = parser.parse_args()


    seeds = [1934,1944,1954,1964, 1974,1984, 1994,2004, 2014,2024]
    # seeds = [2024]
    plot_acc_, AUC_list, AP_list = [], [], []
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for seed in seeds:
        # 设置 PyTorch 随机种子
        torch.manual_seed(seed)
        # 设置 Python 随机种子
        random.seed(seed)
        # 设置 NumPy 随机种子
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        META = meta(args)
        plot_acc, result = META.main(args)
        [confusionmatrix, classification, AUC, AP] = result
        AUC_list.append(AUC)
        AP_list.append(AP)
    plot_acc = np.array(plot_acc)    # iter-acc
    AUC_list = np.array(AUC_list)
    AP_list = np.array(AP_list)
    print('mean_AUC', np.mean(AUC_list))
    print('mean_AP', np.mean(AP_list))
    if len(seeds) > 1:
        np.save('result/plot_acc/'+ args.dataset + '_' +args.exp_name + '_shot' + str(args.example_number+args.query_size) + '.npy', plot_acc)
        np.save('result/plot_AUC/' + args.dataset + '_' +args.exp_name + '_shot' + str(args.example_number+args.query_size) + '.npy', AUC_list)
        np.save('result/plot_AP/' + args.dataset + '_' +args.exp_name + '_shot' + str(args.example_number+args.query_size) + '.npy', AP_list)

    # META = meta(args)
    # META.main(args)
    # META.test(args)
    # META.caculate_metrix(args)
    # META.test_other()
    
    
    # test(model, device, test0_batch, test1_batch)
    