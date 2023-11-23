import os.path as osp

from torch_geometric.datasets import Planetoid, Coauthor
from torch_geometric.transforms import ToSparseTensor, Compose, NormalizeFeatures
from ogb.nodeproppred import PygNodePropPredDataset

from util import index_to_mask, mask_to_index


def get_transform(normalize_features, transform):
    if transform is not None and normalize_features:
        transform = Compose([NormalizeFeatures(), transform])
    elif normalize_features:
        transform = NormalizeFeatures()
    elif transform is not None:
        transform = transform
    return transform


def get_planetoid_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Planetoid(path, name)

    dataset.transform = get_transform(normalize_features, transform)

    return dataset


def get_coauthor_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Coauthor(path, name)
    dataset.transform = get_transform(normalize_features, transform)
    return dataset


def get_ogbn_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = PygNodePropPredDataset(name, path)
    dataset.transform = get_transform(normalize_features, transform)
    return dataset


def get_dataset(dataset_name, normalize_features, sparse=True):
    if sparse:
        transform = ToSparseTensor()
    else:
        transform = None

    if dataset_name.lower() in ['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'ogbn-mag']:
        dataset = get_ogbn_dataset(dataset_name, normalize_features, transform=transform)
        non_transformed_dataset = get_ogbn_dataset(dataset_name, normalize_features)
        data = dataset[0]
        split_idx = dataset.get_idx_split()
        data.train_mask = index_to_mask(split_idx['train'], data.x.shape[0])
        data.test_mask = index_to_mask(split_idx['test'], data.x.shape[0])
        data.val_mask = index_to_mask(split_idx['valid'], data.x.shape[0])
        return non_transformed_dataset, dataset, data, split_idx

    if dataset_name.lower() in ['cora', 'citeseer', 'pubmed']:
        dataset = get_planetoid_dataset(dataset_name, normalize_features, transform=transform)
        non_transformed_dataset = get_planetoid_dataset(dataset_name, normalize_features)
        data = dataset[0]

    elif dataset_name.lower() in ['cs', 'physics']:
        dataset = get_coauthor_dataset(dataset_name, normalize_features, transform=transform)
        non_transformed_dataset = get_coauthor_dataset(dataset_name, normalize_features)
        data = dataset[0]

    split_idx = {'train': mask_to_index(data.train_mask),
                 'valid': mask_to_index(data.val_mask),
                 'test': mask_to_index(data.test_mask)
                 }
    return non_transformed_dataset, dataset, data, split_idx

