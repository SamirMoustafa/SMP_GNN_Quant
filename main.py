import torch
import torch.nn.functional as F
import random
import argparse
import time
import math
import collections
from sklearn.utils import shuffle
from sklearn.model_selection import ParameterGrid
from util import Logger, str2bool
from SMPGNN_Iter import get_model

from train_eval import train, test
from torch.optim import Adam
from torch.optim import lr_scheduler
import numpy as np
import os
import torch
from dataset import get_dataset
def parse_args():
    parser = argparse.ArgumentParser(description='SMP_quant')
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay1', type=float, default=0.0005)
    parser.add_argument('--lr1', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--normalize_features', type=str2bool, default=True)
    parser.add_argument('--random_splits', type=int, default=0, help='default: fix split')

    parser.add_argument('--seed', type=int, default=1234567890)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--lambda1', type=float, default=3)
    parser.add_argument('--lambda2', type=float, default=3)
    parser.add_argument('--prop_mode', type=str, default=None) 
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--qtype', type=str, default=None)
    parser.add_argument("--symmetric", type=bool, default=False)
    parser.add_argument('--BT_mode', type=str, default=None)

    
    parser.add_argument('--embedding_quant', type=bool, default=False)
    
    parser.add_argument('--lambda_threshold', type=float, default=0)
    parser.add_argument('--alpha_threshold', type=float, default=0)
    
    parser.add_argument("--save_model", type=bool, default=False)
 
    args = parser.parse_args()
    args.ogb = False
    return args

def tensor_dim_slice(tensor, dim, dim_slice):
    return tensor[(dim if dim >= 0 else dim + tensor.dim()) * (slice(None), ) + (dim_slice, )]

#@torch.jit.script
def packshape(shape, dim : int = -1, mask : int = 0b00000001, dtype = torch.uint8, pack = True):
    dim = dim if dim >= 0 else dim + len(shape)
    bits, nibble = (8 if dtype is torch.uint8 else 16 if dtype is torch.int16 else 32 if dtype is torch.int32 else 64 if dtype is torch.int64 else 0), (1 if mask == 0b00000001 else 2 if mask == 0b00000011 else 4 if mask == 0b00001111 else 8 if mask == 0b11111111 else 0)
    # bits = torch.iinfo(dtype).bits # does not JIT compile
    assert nibble <= bits and bits % nibble == 0
    nibbles = bits // nibble
    shape = (shape[:dim] + (int(math.ceil(shape[dim] / nibbles)), ) + shape[1 + dim:]) if pack else (shape[:dim] + (shape[dim] * nibbles, ) + shape[1 + dim:])
    return shape, nibbles, nibble

#@torch.jit.script
def packbits(tensor, dim : int = -1, mask : int = 0b00000001, out = None, dtype = torch.uint8):
    dim = dim if dim >= 0 else dim + tensor.dim()
    shape, nibbles, nibble = packshape(tensor.shape, dim = dim, mask = mask, dtype = dtype, pack = True)
    out = out if out is not None else torch.empty(shape, device = tensor.device, dtype = dtype)
    assert out.shape == shape
    
    if tensor.shape[dim] % nibbles == 0:
        shift = torch.arange((nibbles - 1) * nibble, -1, -nibble, dtype = torch.uint8, device = tensor.device)
        shift = shift.view(nibbles, *((1, ) * (tensor.dim() - dim - 1)))
        torch.sum(tensor.view(*tensor.shape[:dim], -1, nibbles, *tensor.shape[1 + dim:]) << shift, dim = 1 + dim, out = out)
        
    else:
        for i in range(nibbles):
            shift = nibble * i
            sliced_input = tensor_dim_slice(tensor, dim, slice(i, None, nibbles))
            sliced_output = out.narrow(dim, 0, sliced_input.shape[dim])
            if shift == 0:
                sliced_output.copy_(sliced_input)
            else:
                sliced_output.bitwise_or_(sliced_input << shift)
    return out

#@torch.jit.script
def unpackbits(tensor, dim : int = -1, mask : int = 0b00000001, shape = None, out = None, dtype = torch.uint8):
    dim = dim if dim >= 0 else dim + tensor.dim()
    shape_, nibbles, nibble = packshape(tensor.shape, dim = dim, mask = mask, dtype = tensor.dtype, pack = False)
    shape = shape if shape is not None else shape_
    out = out if out is not None else torch.empty(shape, device = tensor.device, dtype = dtype)
    assert out.shape == shape
    
    if shape[dim] % nibbles == 0:
        shift = torch.arange((nibbles - 1) * nibble, -1, -nibble, dtype = torch.uint8, device = tensor.device)
        shift = shift.view(nibbles, *((1, ) * (tensor.dim() - dim - 1)))
        return torch.bitwise_and((tensor.unsqueeze(1 + dim) >> shift).view_as(out), mask, out = out)
    
    else:
        for i in range(nibbles):
            shift = nibble * i
            sliced_output = tensor_dim_slice(out, dim, slice(i, None, nibbles))
            sliced_input = tensor.narrow(dim, 0, sliced_output.shape[dim])
            torch.bitwise_and(sliced_input >> shift, mask, out = sliced_output)
    return out  

from collections.abc import Iterable 


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    args = parse_args()
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = args.device   
   
    if args.random_splits > 0:
        random_split_num = args.random_splits
        print(f'random split {random_split_num} times and each for {args.runs} runs')
    else:
        random_split_num = 1
        print(f'fix split and run {args.runs} times')

    logger = Logger(args.runs * random_split_num)

    data_random_split = {}
  
    lambda_threshold = args.lambda_threshold
    alpha_threshold = args.alpha_threshold
    
    split = random_split_num -1
    original_dataset, dataset, data, split_idx = get_dataset(args, split)

    train_idx = split_idx['train']
    data = data.to(device)
    if not isinstance(data.adj_t, torch.Tensor):
        data.adj_t = data.adj_t.to_symmetric()

 
   
    logger = Logger(args.runs * random_split_num)
    test_max_accu = 0.0
    for run in range(args.runs):
        runs_overall = split * args.runs + run
        model = get_model(args, dataset, original_dataset,args.dataset, eta_lambda=lambda_threshold,alpha_threshold=alpha_threshold)
 
        model.reset_parameters()
        alpha_parameter_list = []
        parameter_list = []
      
        for name, para in model.named_parameters():
            if name.find("alpha_")!=-1:
                alpha_parameter_list.append(para)
            else:
                parameter_list.append(para)
         
        optimizer = torch.optim.Adam([
            dict(params=alpha_parameter_list, weight_decay=args.weight_decay,lr=args.lr),
            dict(params=parameter_list, weight_decay=args.weight_decay1,lr=args.lr1)],)
        

        for epoch in range(1, 1 + args.epochs):
            args.current_epoch = epoch
            if args.embedding_quant: 
                loss = train(model, data, train_idx, optimizer,embedding_quant = args.embedding_quant)
                train_acc, valid_acc, test_acc, prop_val = test(model, data, split_idx, embedding_quant = args.embedding_quant)
                result = train_acc, valid_acc, test_acc 
                
            else:
                loss = train(model, data, train_idx, optimizer)
                result = test(model, data, split_idx)
            logger.add_result(runs_overall, result)

            if args.log_steps > 0:
                if epoch % args.log_steps == 0:
                    train_acc, valid_acc, test_acc = result
                if epoch % 1 == 0:
                    print(f'Split: {split + 1:02d}, 'f'Run: {run + 1:02d}, 'f'Epoch: {epoch:02d}, 'f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_acc:.2f}%, 'f'Valid: {100 * valid_acc:.2f}% 'f'Test: {100 * test_acc:.2f}%')
                if test_max_accu < test_acc:
                    test_max_accu = test_acc
                    if args.embedding_quant:
                        read_data = prop_val[0].detach()
                        scale = prop_val[1].detach()
                        zero_point = prop_val[2].detach()
                        quant_embedding = ((read_data)*(1/scale)+zero_point).round().numpy().astype(np.uint8)
                        quant_embedding = torch.from_numpy(quant_embedding)
                        dtype = torch.uint8
                        if args.qtype == "INT2":
                            nbits = 2    
                        if args.qtype == "INT4":
                            nbits = 4
                        if args.qtype == "INT8":
                            nbits = 8
                        mask = (1 << nbits) - 1
                        bit_reprsentation = packbits(read_data, mask = mask)

                        embedding_store_path = args.dataset + "_" + str(args.qtype) + "_store_embedding.plt"
                        torch.save(quant_embedding, embedding_store_path)
                        print("save quantized embedding to:"+str(embedding_store_path))


            if epoch % 50 == 0:
                optimizer.param_groups[1]["lr"] = 0.7* optimizer.param_groups[1]["lr"]


        if args.log_steps > 0:
            print(print(f'Split: {split + 1:02d}, 'f'Run: {run + 1:02d}'))
            store_path = None#args.store_path
            logger.print_statistics(runs_overall, store_path)
    
    
    
    
    model_orderDict = collections.OrderedDict()
    if args.save_model:
        if args.qtype is  None:
            torch.save(model.state_dict(),PATH)
        else:
            model_orderDict = model.state_dict()

            for i in range(9):
                name = "prop."+str(i+1)+".alpha_message"
                model_orderDict[name] = model.prop.messagegroup_quantizers['prop'][i]['message'].custom_alpha.detach()

                name = "prop."+str(i+1)+".alpha_aggregate"
                model_orderDict[name] = model.prop.messagegroup_quantizers['prop'][i]['aggregate'].custom_alpha.detach()

                name = "prop."+str(i+1)+".alpha_update"
                model_orderDict[name] = model.prop.messagegroup_quantizers['prop'][i]['update_q'].custom_alpha.detach()

            model_orderDict['lin1.weight'] = {}
            model_orderDict['lin2.weight'] = {}
            model_orderDict['merge_linearquant.lin1.weight'] = {}
            model_orderDict['merge_linearquant.lin2.weight'] = {}
            del model_orderDict['lin1.weight']
            del model_orderDict['lin2.weight']
            del model_orderDict['merge_linearquant.lin1.weight']
            del model_orderDict['merge_linearquant.lin2.weight']
            if args.qtype == "INT8":
                model_orderDict['lin1.weight'] = model.merge_linearquant.weight1_int.byte()#.numpy()
                model_orderDict['lin1.weight1_scale'] = model.merge_linearquant.weight1_scale
                model_orderDict['lin1.weight1_zeropoint'] = model.merge_linearquant.weight1_zeropoint

                model_orderDict['lin2.weight'] = model.merge_linearquant.weight2_int.byte()#.numpy()
                model_orderDict['lin2.weight2_scale'] = model.merge_linearquant.weight2_scale
                model_orderDict['lin2.weight2_zeropoint'] = model.merge_linearquant.weight2_zeropoint
            elif args.qtype == "INT4":
                nibble = 4

                mask = (1 << nibble) - 1

                weight1_int = packbits(model.merge_linearquant.weight1_int.byte(), mask = mask)
                weight2_int = packbits(model.merge_linearquant.weight2_int.byte(), mask = mask)

                model_orderDict['lin1.weight'] = weight1_int#.byte()
                model_orderDict['lin1.weight1_scale'] = model.merge_linearquant.weight1_scale
                model_orderDict['lin1.weight1_zeropoint'] = model.merge_linearquant.weight1_zeropoint

                model_orderDict['lin2.weight'] = weight2_int#.byte()#.numpy()
                model_orderDict['lin2.weight2_scale'] = model.merge_linearquant.weight2_scale
                model_orderDict['lin2.weight2_zeropoint'] = model.merge_linearquant.weight2_zeropoint

            PATH = args.dataset + "_" + str(args.qtype) + "_modelsize.plt"
            print("save model to:"+str(PATH))
            torch.save(model_orderDict,PATH)

if __name__ == "__main__":
    main()
