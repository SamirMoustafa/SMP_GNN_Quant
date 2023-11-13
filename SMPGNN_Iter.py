import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.utils import add_self_loops, degree
# from message_propagation_fun import prop_part
from message_propagation_fun import prop_part_QUANT
from scipy.stats import kurtosis
from scipy.stats import skew
from linear_quantization import LinearQuantized

from QLR_quantizer_fun import QLRQuantizer


def make_quantizers(qypte, datasetName, prop_mode, BT_mode=None):
   
    layer_quantizers = {
        "inputs": create_messagequantizer(qypte, datasetName, prop_mode, BT_mode = BT_mode),
        "weights": create_messagequantizer(qypte, datasetName, prop_mode, BT_mode = BT_mode),
        "features": create_messagequantizer(qypte, datasetName, prop_mode, BT_mode = BT_mode),
        "norm": create_messagequantizer(qypte, datasetName, prop_mode, BT_mode = BT_mode),
    }
    return layer_quantizers



def make_messagequantizers(qypte, datasetName, prop_mode, BT_mode=None):
   
    layer_quantizers = {
        "message": create_messagequantizer(qypte, datasetName, prop_mode, BT_mode = BT_mode),
        "aggregate": create_messagequantizer(qypte, datasetName, prop_mode, BT_mode = BT_mode),
        "aggregate": create_messagequantizer(qypte, datasetName, prop_mode, BT_mode = BT_mode),
        "update_q": create_messagequantizer(qypte, datasetName, prop_mode, BT_mode = BT_mode),
    }
    return layer_quantizers


def create_messagequantizer(qypte, datasetName, prop_mode, BT_mode=None):
    if qypte == "INT4":
        bit_num = 4
    elif qypte == "INT2":
        bit_num = 2
    elif qypte == "INT8":
        bit_num = 8
    elif qypte == "INT1":
        bit_num = 1
    elif qypte == "FP32":
        bit_num = 32    
    return QLRQuantizer(bit_num, datasetName, prop_mode, BT_mode = BT_mode)  
    
    
def make_finalquantizers(qypte, datasetName, prop_mode, BT_mode=None):
   
    layer_quantizers = {
        "out": create_messagequantizer(qypte, datasetName, prop_mode, BT_mode = BT_mode),}
    return layer_quantizers   


class SMPGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, prop, qtype, datasetName=None, prop_mode=None, BT_mode =None, embedding_quant=None, **kwargs):
        super(SMPGNN, self).__init__()
        
        self.dropout = dropout
        self.prop = prop
        self.qtype = qtype
        self.embedding_quant = embedding_quant
        
        lin1_quantizers = make_quantizers(qtype, datasetName, prop_mode, BT_mode = BT_mode)
        lin2_quantizers = make_quantizers(qtype, datasetName, prop_mode, BT_mode = BT_mode)
        self.lin1 = LinearQuantized(in_channels, hidden_channels,lin1_quantizers,1.0)
        self.lin2 = LinearQuantized(hidden_channels, out_channels,lin2_quantizers,1.0)
            
        self.reg_params1 = list(self.lin1.parameters()) 
        self.reg_params2 = list(self.lin2.parameters()) 

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, data):
        x, adj_t, = data.x, data.adj_t
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = F.relu(self.lin1(x, training=self.training)[0])
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        if not self.embedding_quant:
            
            x = self.lin2(x, training=self.training)[0]
            x = self.prop(x, adj_t, data=data)
        else:
            
            x = self.prop(x, adj_t, data=data)
            prop_val = self.lin2(x, training=self.training)
            x = prop_val[0]
            return F.log_softmax(x, dim=1), prop_val
#             save the quantized embedding for similarity search
        
        return F.log_softmax(x, dim=1)



def get_model(args, dataset,original_dataset,datasetName,eta_lambda=None,alpha_threshold=None ):

    edge_index = original_dataset[0].edge_index
    edge_index, _ = add_self_loops(edge_index, num_nodes=original_dataset[0].x.size(0))

    data = dataset[0]
     

    Model = SMPGNN
    quantizers_list = {}
    quantizers_list['prop'] = {}
    quantizers_list['gap_prop'] = {}
        
    for i in range(args.K):
        quantizers = make_messagequantizers(args.qtype, datasetName, args.prop_mode, BT_mode =args.BT_mode)
        quantizers_list['prop'][i] = quantizers
        quantizers = make_messagequantizers(args.qtype, datasetName, args.prop_mode, BT_mode =args.BT_mode)
        quantizers_list['gap_prop'][i] = quantizers
    out_quantizer = make_finalquantizers(args.qtype, datasetName, args.prop_mode, BT_mode =args.BT_mode)
    prop =  prop_part_QUANT(K=args.K, lambda1=args.lambda1, lambda2=args.lambda2, cached=True, oriedge_index=edge_index,prop_mode=args.prop_mode, dataName=datasetName, qtype = args.qtype, messagegroup_quantizers=quantizers_list, eta_lambda=eta_lambda, alpha_threshold=alpha_threshold, finalout_quantizer=out_quantizer, embedding_quant=args.embedding_quant)
        
    model = Model(in_channels=data.num_features, hidden_channels=args.hidden_channels, out_channels=dataset.num_classes, 
                       dropout=args.dropout, prop=prop, qtype = args.qtype, datasetName=datasetName,prop_mode=args.prop_mode, BT_mode =args.BT_mode, embedding_quant=args.embedding_quant)

    return model