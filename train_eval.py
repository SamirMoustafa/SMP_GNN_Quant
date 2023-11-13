import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator
criterion = torch.nn.CrossEntropyLoss()

def train(model, data, train_idx, optimizer, embedding_quant=None):
    model.train()
    optimizer.zero_grad()
    if embedding_quant: 
        out = model(data=data)[0][train_idx]
    else:
        out = model(data=data)[train_idx]
    if len(data.y.shape) == 1:
        y = data.y[train_idx]
    else:
        y = data.y.squeeze(1)[train_idx]  ## for ogb data
    loss = F.nll_loss(out, y)

    loss.backward(retain_graph=True)
    optimizer.step()
    return loss.item()
 
@torch.no_grad()
def test(model, data, split_idx, embedding_quant=None):
    model.eval()
    
    if embedding_quant:
        out, prop_val = model(data=data)
    else:
        out = model(data=data)
        
    y_pred = out.argmax(dim=-1, keepdim=True)

    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1) # for non ogb datas
    else:
        y = data.y

    evaluator = Evaluator(name='ogbn-arxiv')
    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    
    if embedding_quant:
        return train_acc, valid_acc, test_acc, prop_val
    else:
        return train_acc, valid_acc, test_acc