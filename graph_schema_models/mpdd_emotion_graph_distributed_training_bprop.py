#!/usr/bin/env python
# coding: utf-8

# Import required utility packages
import os
import sys
from datetime import datetime
import time
import numpy as np
import random
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import pandas as pd
from pandas import DataFrame as DF
from tqdm.auto import tqdm
import logging
from tabulate import tabulate

# Set environment variable for distributed processing
os.environ["MKL_SERVICE_FORCE_INTEL"] = '1'
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# Import torch and transformers packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.algorithms.join import Join
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.join import Join

from transformers import AutoModel

# Import DGL packages
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.graphbolt as gb
import dgl.nn as dglnn
import dgl.data
from dgl.nn.pytorch import SAGEConv


# Reproducibility parameters
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
dgl.seed(4056)
torch.manual_seed(4056)
torch.cuda.manual_seed(4056)
torch.backends.cudnn.benchmarks = False
torch.backends.cudnn.deterministic = True
np.random.seed(4056)
random.seed(4056)

# Set default device for computing
device = torch.device('cuda:0')


# ### Utils

# Data paths and parameter save paths:
fpath = '/home/rpujari/gilbreth_scratch/scratch_ml/DARPA/'
mpdd_path = '/home/rpujari/gilbreth_scratch/scratch_ml/DARPA/mpdd/'
base_dir = '/home/rpujari/gilbreth_scratch/scratch_ml/DARPA/mpdd/saved_datasets/mpdd_emotion_dgl_ondisk'
save_path = '/home/rpujari/gilbreth_scratch/scratch_ml/DARPA/ta2snapshot_saved_parameters/'
log_path = '/home/rpujari/gilbreth_scratch/scratch_ml/DARPA/training_logs/'
logging.basicConfig(filename=log_path + 'mpdd_emotions_' + str(datetime.now()).replace(' ', '_') + '.log', level=logging.DEBUG)

en_model_name = 'distilroberta-base'
# en_model_name = 'roberta-base'
zh_model_name = 'hfl/chinese-roberta-wwm-ext'

emotion_label_list = ['angry', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

#Define graph model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, encoder_bprop=True):
        super(GraphSAGE, self).__init__()
        self.en_encoder = AutoModel.from_pretrained(en_model_name)
        self.en_encoder_bprop = AutoModel.from_pretrained(en_model_name)
        self.zh_encoder = AutoModel.from_pretrained(zh_model_name)
        self.zh_encoder_bprop = AutoModel.from_pretrained(zh_model_name)
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(h_feats, num_classes, aggregator_type='mean')
        
        # self.conv_mid_1 = SAGEConv(h_feats, h_feats, aggregator_type='mean')
        # self.conv_mid_2 = SAGEConv(h_feats, h_feats, aggregator_type='mean')
        # self.conv_mid_3 = SAGEConv(h_feats, h_feats, aggregator_type='mean')
        # self.conv_mid_4 = SAGEConv(h_feats, h_feats, aggregator_type='mean')

        for p in self.en_encoder.parameters():
            p.requires_grad = False
        for p in self.zh_encoder.parameters():
            p.requires_grad = False
        if encoder_bprop == False:
            for p in self.en_encoder_bprop.parameters():
                p.requires_grad = False
            for p in self.zh_encoder_bprop.parameters():
                p.requires_grad = False
                
        # Initialize the GraphSAGE layers randomly
        self._initialize_weights()

    def _initialize_weights(self):
        """Randomly initialize the weights of the DGL SAGEConv layers."""
        for layer in [self.conv1, self.conv2]:
            nn.init.xavier_uniform_(layer.fc_neigh.weight)  # Xavier initialization for the weight matrix
            nn.init.xavier_uniform_(layer.fc_self.weight)  # Xavier initialization for the weight matrix
            if layer.fc_self.bias is not None:
                nn.init.zeros_(layer.fc_self.bias)  # Set bias to zeros
                
    def forward(self, blocks, input_ids, attention_mask, langs):
        num_nodes = langs.size(0)
        
        sorted_ids = sorted(list(range(num_nodes)), key=lambda x: langs[x])
        remap_ids = [sorted_ids.index(i) for i in range(num_nodes)]
        
        sorted_input_ids = input_ids[sorted_ids, :].to(input_ids.device)
        sorted_attention_mask = attention_mask[sorted_ids, :].to(attention_mask.device)
        
        boundaries = []
        data_langs = []
        prev_lang = -1
        for i in range(num_nodes):
            if langs[sorted_ids[i]] != prev_lang:
                boundaries.append(i)
                prev_lang = langs[sorted_ids[i]]
                data_langs.append(langs[sorted_ids[i]])
        boundaries.append(num_nodes)
        
        in_feats = []
        i = 0
        for b, e in zip(boundaries[:-1], boundaries[1:]):
            if int(data_langs[i]) == 0:
                in_feat = self.en_encoder(input_ids=sorted_input_ids[b:e, :], attention_mask=sorted_attention_mask[b:e, :]).last_hidden_state[:, 0, :].contiguous()
            elif int(data_langs[i]) == 1:
                in_feat = self.en_encoder_bprop(input_ids=sorted_input_ids[b:e, :], attention_mask=sorted_attention_mask[b:e, :]).last_hidden_state[:, 0, :].contiguous()
            elif int(data_langs[i]) == 2:
                in_feat = self.zh_encoder(input_ids=sorted_input_ids[b:e, :], attention_mask=sorted_attention_mask[b:e, :]).last_hidden_state[:, 0, :].contiguous()
            elif int(data_langs[i]) == 3:
                in_feat = self.zh_encoder_bprop(input_ids=sorted_input_ids[b:e, :], attention_mask=sorted_attention_mask[b:e, :]).last_hidden_state[:, 0, :].contiguous()
            in_feats.append(in_feat)
            i += 1
        
        in_feats = torch.cat(in_feats, dim=0).contiguous()
        in_feats = in_feats[remap_ids, :].contiguous()
        
        block_num = 0
        h = self.conv1(blocks[block_num], in_feats)
        h = F.relu(h)
        block_num += 1

        # h = self.conv_mid_1(blocks[block_num], h)
        # h = F.relu(h)
        # block_num += 1

        # h = self.conv_mid_2(blocks[block_num], h)
        # h = F.relu(h)
        # block_num += 1
        
#         h = self.conv_mid_3(blocks[block_num], h)
#         h = F.relu(h)
#         block_num += 1
        
#         h = self.conv_mid_4(blocks[block_num], h)
#         h = F.relu(h)
#         block_num += 1
        
        h = self.conv2(blocks[block_num], h)
        # h = F.relu(h)
        
        return h.squeeze(1)
        

def create_dataloader(
    graph,
    features,
    itemset,
    device,
    is_train,
    bsz=15
):
    datapipe = gb.DistributedItemSampler(
        item_set=itemset,
        batch_size=bsz,
        drop_last=True,#is_train,
        shuffle=False,#is_train
        drop_uneven_inputs=True,#is_train,
    )
    datapipe = datapipe.copy_to(device)
    # Now that we have moved to device, sample_neighbor and fetch_feature steps will be executed on GPUs.
    # Full neighbor sampling
    datapipe = datapipe.sample_neighbor(graph, [5, 5])
    datapipe = datapipe.fetch_feature(features, node_feature_keys=["input_ids", "attention_mask", "language"])
    return gb.DataLoader(datapipe)


def weighted_reduce(tensor, weight, dst=0):
    rank = dist.get_rank()
    dist.reduce(tensor=tensor, dst=dst)
    weight = torch.tensor(weight, device=tensor.device)
    dist.reduce(tensor=weight, dst=dst)
    return tensor / weight

def gather_outputs(tensor, tensor_list=None, dst=0):
    rank = dist.get_rank()
    if rank == dst:
        dist.gather(tensor=tensor, gather_list=tensor_list, dst=dst)
    else:
        dist.gather(tensor=tensor, dst=dst)

@torch.no_grad()
def evaluate(rank, model, graph, features, itemset, num_classes, device, bsz=30):
    # print(rank, itemset)
    with torch.no_grad():
        dataloader = create_dataloader(
            graph,
            features,
            itemset,
            device,
            is_train=False,
            bsz=bsz
        )

        model.eval()
        y = []
        y_hats = []
        ids = []
        logits = []

        # print('Starting Evaluation', rank)
        bnum = 0
        for data in dataloader:
            blocks = data.blocks
            input_ids = data.node_features["input_ids"]
            attention_mask = data.node_features["attention_mask"]
            language = data.node_features["language"]
            logit_ = model(blocks, input_ids, attention_mask, language)
            preds = logit_.argmax(1)
            y.append(data.labels)
            y_hats.append(preds)
            ids.append(data.seeds)
            logits.append(logit_)
            bnum += 1

        Y = torch.cat(y, dim=0).to(device)
        Y_hat = torch.cat(y_hats, dim=0).to(device)
        Logits = torch.cat(logits, dim=0).to(device)
        IDs = torch.cat(ids, dim=0).to(device)

        return Y, Y_hat, Logits, IDs

@torch.no_grad()
def generate_result(
    world_size,
    rank,
    model,
    graph,
    features,
    itemset,
    num_classes,
    device,
    bsz=30
):
    # Test the model.
    if rank == 0:
        print("Testing...")

    test_set_sample = gb.ItemSet(itemset[:], names=itemset.names)
    
    Y, Y_hat, Logits, IDs = evaluate(
        rank,
        model,
        graph,
        features,
        test_set_sample,
        num_classes,
        device,
        bsz=bsz
    )

    Y_size = torch.tensor([Y.size(0)], device=device)
    Y_sizes_all = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    # print(Y_size, rank)
    gather_outputs(Y_size, Y_sizes_all, dst=0)
    # if rank == 0:
    #     print(Y_sizes_all)
    
    if rank == 0:
        # Allocate tensors based on gathered sizes
        Y_all = [torch.zeros(s.item(), dtype=torch.long, device=device) for s in Y_sizes_all]
        Y_hat_all = [torch.zeros(s.item(), dtype=torch.long, device=device) for s in Y_sizes_all]
        Logits_all = [torch.zeros((s.item(), num_classes), dtype=torch.float, device=device) for s in Y_sizes_all]
        IDs_all = [torch.zeros(s.item(), dtype=torch.int, device=device) for s in Y_sizes_all]

        dist.barrier()
        
        # print(Y.size(), rank)
        gather_outputs(Y, Y_all, dst=0)
        gather_outputs(Y_hat, Y_hat_all, dst=0)
        gather_outputs(Logits, Logits_all, dst=0)
        gather_outputs(IDs, IDs_all, dst=0)
    else:
        # print('Barrier', rank)
        dist.barrier()
    
        # print(Y.size(), rank)
        gather_outputs(Y, dst=0)
        gather_outputs(Y_hat, dst=0)
        gather_outputs(Logits, dst=0)
        gather_outputs(IDs, dst=0)

    if rank == 0:
        Y_all_t = torch.cat(Y_all, dim=0).cpu().numpy()
        Y_hat_all_t = torch.cat(Y_hat_all, dim=0).cpu().numpy()

        report = classification_report(Y_all_t, Y_hat_all_t, labels=list(range(num_classes)), target_names=emotion_label_list, zero_division=0, output_dict=True)
        report_df = DF(report).transpose()

        # Compute accuracy on validation
        if 'accuracy' in report:
            test_acc = report['accuracy']
        elif 'micro avg' in report:
            test_acc = report['micro avg']['f1-score']
        else:
            test_acc = -1
        
        test_f1 = report['weighted avg']['f1-score']

        print(f"Test Accuracy {test_acc:.4f}")
        print(f"Test F1 {test_f1:.4f}")
        print(tabulate(report_df, headers='keys', tablefmt='psql'))
        return test_f1, test_acc


def train(
    world_size,
    rank,
    graph,
    features,
    train_set,
    valid_set,
    num_classes,
    label_count,
    model,
    optimizer,
    device,
    model_path='model.pt',
    num_epochs=10,
    bsz=10,
):
    model_save_path = save_path + model_path
    optimizer_save_path = save_path + model_path[:-3] + '_optimizer.pt'
    
    # Create training data loader.
    dataloader = create_dataloader(
        graph,
        features,
        train_set,
        device,
        is_train=True,
        bsz=bsz
    )
    
    best_performance = -1

    #Testing before training start
    ret = generate_result(
        world_size,
        rank,
        model,
        graph,
        features,
        valid_set,
        num_classes,
        device,
        bsz=bsz
    )
    if ret:
        best_performance = ret[0]
        print('Updating best performance to', best_performance)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()

        model.train()
        total_loss = torch.tensor(0, dtype=torch.float, device=device)
        num_train_items = 0
        # print('Total-loss:', total_loss)
        with Join([model]):
            bnum = 0
            for data in dataloader:
                # The input features are from the source nodes in the first
                # layer's computation graph.
                blocks = data.blocks
                input_ids = data.node_features["input_ids"]
                attention_mask = data.node_features["attention_mask"]
                language = data.node_features["language"]
                logit_ = model(blocks, input_ids, attention_mask, language)
                y = data.labels

                loss_weight = 1 / (label_count + 1)
                loss_weight = loss_weight / torch.sum(loss_weight)
                # Compute loss.
                loss = F.cross_entropy(logit_, y, weight=loss_weight.to(device))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                bnum += 1

                # print('Rank-', rank, 'Batch-', bnum, 'Loss-', loss)
                total_loss += loss.detach() * y.size(0)
                num_train_items += y.size(0)
            

        # Evaluate the model.
        if rank == 0:
            print("Validating...")
        Y, Y_hat, Logits, IDs = evaluate(
            rank,
            model,
            graph,
            features,
            valid_set,
            num_classes,
            device,
            bsz=bsz
        )

        # print('Returned Evaluation', rank)
        # print('Rank loss:', total_loss, num_train_items, total_loss/num_train_items, rank)
        # total_loss = weighted_reduce(total_loss, num_train_items)
        # print('Total loss:', total_loss)

        # print('Returned-Evaluation', rank)
        Y_size = torch.tensor([Y.size(0)], device=device)
        Y_sizes_all = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        # print('Gathering sizes', rank, 'Sending', Y_size)
        # print('Expecting', Y_sizes_all, rank)
        gather_outputs(Y_size, Y_sizes_all, dst=0)
        
        if rank == 0:
            # Print gathered sizes for debugging
            # print("Gathered sizes:", [s.item() for s in Y_sizes_all])
            
            # Allocate tensors based on gathered sizes
            Y_all = [torch.zeros(s.item(), dtype=torch.long, device=device) for s in Y_sizes_all]
            Y_hat_all = [torch.zeros(s.item(), dtype=torch.long, device=device) for s in Y_sizes_all]
            Logits_all = [torch.zeros((s.item(), num_classes), dtype=torch.float, device=device) for s in Y_sizes_all]
            IDs_all = [torch.zeros(s.item(), dtype=torch.int, device=device) for s in Y_sizes_all]
            
            # print('Barrier', rank)
            dist.barrier()
            
            gather_outputs(Y, Y_all, dst=0)
            gather_outputs(Y_hat, Y_hat_all, dst=0)
            gather_outputs(Logits, Logits_all, dst=0)
            gather_outputs(IDs, IDs_all, dst=0)
            # print("Returned gathering", rank)
        else:
            # print('Barrier', rank)
            dist.barrier()
        
            gather_outputs(Y, dst=0)
            gather_outputs(Y_hat, dst=0)
            gather_outputs(Logits, dst=0)
            gather_outputs(IDs, dst=0)
            # print("Returned gathering", rank)

        if rank == 0:
            # print("Calculating performance")
            Y_all_t = torch.cat(Y_all, dim=0).cpu().numpy()
            Y_hat_all_t = torch.cat(Y_hat_all, dim=0).cpu().numpy()
    
            report = classification_report(Y_all_t, Y_hat_all_t, labels=list(range(num_classes)), target_names=emotion_label_list, zero_division=0, output_dict=True)
            report_df = DF(report).transpose()

            # print("Done with calculation")
            # Compute accuracy on validation
            if 'accuracy' in report:
                val_acc = report['accuracy']
            elif 'micro avg' in report:
                val_acc = report['micro avg']['f1-score']
            else:
                val_acc = -1
            
            val_f1 = report['weighted avg']['f1-score']

            print(tabulate(report_df, headers='keys', tablefmt='psql'))

            # Save the best validation accuracy and the corresponding test accuracy.
            if val_f1 >= best_performance:
                print('Saving new parameters')
                best_performance = val_f1
                torch.save(model.state_dict(), model_save_path)
                torch.save(optimizer.state_dict(), optimizer_save_path)
                torch.cuda.empty_cache()
                
        
        # We synchronize before measuring the epoch time.
        # print('Syncing...')
        torch.cuda.synchronize()
        # print('Synced.')
        epoch_end = time.time()
        if rank == 0:
            print(
                f"Epoch {epoch:05d} | "
                f"Average Loss {total_loss.item():.4f} | "
                f"Val Acc {val_acc:.4f} | "
                f"Weighted F1 {val_f1:.4f} | "
                f"Time {epoch_end - epoch_start:.4f}"
            )
        dist.barrier()
            
            
def run(rank, world_size, devices, dataset, model_path='model.pt', init_path=None, resume_training=False, num_epochs=10, bsz=10, lr=1e-5, inference_mode=False, encoder_bprop=False):
    # Set up multiprocessing environment.
    device = devices[rank]
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",  # Use NCCL backend for distributed GPU training
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    # Pin the graph and features in-place to enable GPU access.
    graph = dataset.graph.pin_memory_()
    features = dataset.feature.pin_memory_()
    train_set = dataset.tasks[0].train_set
    valid_set = dataset.tasks[0].validation_set

    train_set_sample = gb.ItemSet(train_set[:], names=train_set.names)
    valid_set_sample = gb.ItemSet(valid_set[:], names=valid_set.names)
    
    num_classes = dataset.tasks[0].metadata["num_classes"]
    label_count = torch.bincount(train_set_sample[:][1])
    print(label_count)

    in_size = 768
    hidden_size = 100
    out_size = num_classes

    # Create GraphSAGE model. It should be copied onto a GPU as a replica.
    model = GraphSAGE(in_size, hidden_size, out_size, encoder_bprop=encoder_bprop).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model_save_path = save_path + model_path
    optimizer_save_path = save_path + model_path[:-3] + '_optimizer.pt'
    
    if resume_training and os.path.exists(model_save_path) and os.path.exists(optimizer_save_path):
        try:
            optimizer_state_dict = torch.load(optimizer_save_path, map_location=device)
            optimizer.load_state_dict(optimizer_state_dict)
            try:
                model_state_dict = torch.load(model_save_path, map_location=device)
                model_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in model_state_dict.items()}
                model.load_state_dict(model_state_dict)
                model = model.to(device)
                # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
                print('Resuming traning from a checkpoint')
            except:
                print('Error 2:', sys.exc_info()[1])
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
                print('Training a fresh model')
        except:
            print('Error 1:', sys.exc_info()[1])
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            print('Training a fresh model')
    elif (not resume_training) and init_path:
        model_state_dict = torch.load(save_path + init_path, map_location=device)
        model_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in model_state_dict.items()}
        model.load_state_dict(model_state_dict)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        print('Training a model from an initalization')
    else:
        print('Training a fresh model')
    

    model = DDP(model, find_unused_parameters=encoder_bprop)
    
    # Model training.
    if not inference_mode:
        if rank == 0:
            print("Training...")
        train(
            world_size,
            rank,
            graph,
            features,
            train_set_sample,
            valid_set_sample,
            num_classes,
            label_count,
            model,
            optimizer,
            device,
            model_path=model_path,
            num_epochs=num_epochs,
            bsz=bsz,
        )
    if inference_mode:
        if rank == 0:
            print('Validation Set Results:')
        ret = generate_result(
            world_size,
            rank,
            model,
            graph,
            features,
            dataset.tasks[0].validation_set,
            num_classes,
            device,
            bsz=1
        )
    if rank == 0:
        print('Test Set Results:')
    ret = generate_result(
        world_size,
        rank,
        model,
        graph,
        features,
        dataset.tasks[0].test_set,
        num_classes,
        device,
        bsz=1
    )



######################################################################
# Spawning Trainer Processes
# --------------------------
#
# The following code spawns a process for each GPU and calls the ``run``
# function defined above.
#


def main():
    if not torch.cuda.is_available():
        print("No GPU found!")
        return

    devices = [
        torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())
    ]
    world_size = len(devices)

    print(f"Training with {world_size} gpus.")

    # Load and preprocess dataset.
    dataset = gb.OnDiskDataset(base_dir).load()

    # Thread limiting to avoid resource competition.
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // world_size)

    mp.set_sharing_strategy("file_system")
    mp.spawn(
        run,
        args=(world_size, devices, dataset,\
              'schema_roberta_mpdd_emotion/distilbert_graphsage_graphbolt_2L_full_bprop.pt',\
              # model_path
              None,\
              # 'schema_roberta_mpdd_emotion/distilbert_graphsage_500_graphbolt_nobprop_4layers_allsampling.pt',\
              # init_path
              True, 50, 5, 1e-5, True, True),
              # resume, epochs, bsz, lr, inference_mode, bprop
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()

