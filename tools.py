import datetime
import torch
import numpy as np
import numpy_indexed as npi


def get_current_time_string():
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    return time_str


def aggragate_torch(value_array, regions, mode='mean'):
    aggr_method = eval('torch.'+mode)
    sorter = torch.argsort(regions.ravel())
    # could be optimised...
    _, inverse_sorter = np.unique(sorter, return_index=True)
    regions_sort = regions.ravel()[sorter]
    value_array_sort = value_array.ravel()[sorter]
    # print(value_array_sort)
    marker_idx = torch.where(torch.diff(regions_sort) == 1)[0]+1
    reduceat_idx = torch.cat(
        [torch.tensor([0]), marker_idx, torch.tensor([regions.numel()])])
    group_counts = reduceat_idx[1:] - reduceat_idx[:-1]
    vs = torch.zeros(len(group_counts)).cuda()
    start = 0
    for i, length in enumerate(group_counts):
        end = start + length
        # torch.mean(value_array_sort[start:end])
        vs[i] = aggr_method(value_array_sort[start:end])
        # vs[i] = torch.sum(value_array_sort[start:end])
        start = end
    return vs.squeeze(-1)


def group_aggregation_(value_array, regions, keep_shape=False, method='mean'):
    groupby = npi.group_by(regions.ravel())
    if method == 'mean':
        keys, values = groupby.mean(value_array.ravel())
    elif method == 'sum':
        keys, values = groupby.sum(value_array.ravel())
    if keep_shape:
        return values[groupby.inverse].reshape(regions.shape)
    else:
        return keys, values


def group_aggregation(value_arrays, regions, keep_shape=False):
    values = []
    for i in range(value_arrays.shape[0]):
        value_array = value_arrays[i]
        k, value = group_aggregation_(
            value_array, regions, keep_shape=keep_shape)
        # k = k[1:]
        # value = value[1:]
        values.append(value)
    values = np.array(values).T
    return k, values
