import time
from models import *
from osgeo import gdal
import numpy as np
import numpy_indexed as npi
from sklearn import metrics
import numpy as np
# from sklearnex import patch_sklearn, unpatch_sklearn
# from sklearn.externals import joblib
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='这是程序的描述信息')

# 添加参数
parser.add_argument('--ckp', type=str, default='None', help='模型文件的路径')
parser.add_argument('--ep', type=int, default=50000, help='训练的轮数')
parser.add_argument('--mode', type=str, choices=['train', 'map','eval','feature'], default='train', help='程序的模式')
parser.add_argument('--type', type=str, default='both', help='损失函数')
parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
parser.add_argument('--dis','-d',type=str,default='',help='额外描述')
parser.add_argument('--year','-y',type=str,default='2016',help='数据年份')
parser.add_argument('--savepath','-s',type=str,default='None')
parser.add_argument('--rfac',type=str,default='None',help='移除的要素')
# 解析参数
args = parser.parse_args()
args.map = "F:/data_lib/00_HK_Population_source_data/"+{'2021':"HongKong2021_FeatureCube.tif",'2016':'HongKong2016_FeatureCube.tif'}[args.year]
if (args.mode == 'map') & (args.savepath == 'None'):
    raise ValueError('Please set the save path.')
# 使用参数
print("ckp 文件路径：", args.ckp,end='')
print("训练轮数：", args.ep,end='')
print("程序模式：", args.mode,end='')
print("数据类型：", args.type,end='')
print("学习率：", args.lr,end='')
print("额外描述：",args.dis,end='')
print("年份：",args.year,end='')
print("移除要素：",args.rfac)
def get_current_time_string():
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    return time_str


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


# load and split data HK
Dataset = gdal.Open(
    args.map).ReadAsArray()
factors_index = {'RSI':[0,1,2,3,4],
                 'Rd':[5,6],
                 'Topo':[7,8],
                 'Bui':[9,10],
                 'LU':[11,12,13,14,15,16,17,18,19,20,21],
                 'None':[]}
rfac = args.rfac.split('-')
rfac_index = []
for i in rfac:
    rfac_index+=factors_index[i]

# factors = tuple(factors)
DataFactors = Dataset[:-6]
DataFactors[DataFactors>1e30] = 0
DataFactors[DataFactors <= -1e38] = 0
PopDensity = Dataset[(-4, -1), :, :]
Area = Dataset[(-5, -2), :, :]
RegionMask = Dataset[(-6, -3), :, :]
c, m, n = DataFactors.shape

# norm and stan
# the pixel out HK will be ignored in mean and std
pre_type = ['None', 'Stan', 'Norm', 'N&S'][-1]
if pre_type == 'None':
    Norm_DataFactors = DataFactors
elif pre_type == 'Norm':
    mean_list = DataFactors[:, RegionMask[0] > 0].mean(1, keepdims=True)
    mean_list = mean_list.reshape(mean_list.shape[0], mean_list.shape[1], 1)
    std_list = DataFactors[:, RegionMask[0] > 0].std(1, keepdims=True)
    std_list = std_list.reshape(std_list.shape[0], std_list.shape[1], 1)
    Norm_DataFactors = (DataFactors-mean_list)/std_list
elif pre_type == 'Stan':
    max_list = DataFactors[:, RegionMask[0] > 0].max(1, keepdims=True)
    max_list = max_list.reshape(max_list.shape[0], max_list.shape[1], 1)
    min_list = DataFactors[:, RegionMask[0] > 0].min(1, keepdims=True)
    min_list = min_list.reshape(min_list.shape[0], min_list.shape[1], 1)
    Norm_DataFactors = (DataFactors - min_list)/(max_list-min_list)
else:
    max_list = DataFactors[:, RegionMask[0] > 0].max(1, keepdims=True)
    max_list = max_list.reshape(max_list.shape[0], max_list.shape[1], 1)
    min_list = DataFactors[:, RegionMask[0] > 0].min(1, keepdims=True)
    min_list = min_list.reshape(min_list.shape[0], min_list.shape[1], 1)
    Standard_DataFactors = (DataFactors - min_list)/(max_list-min_list)
    mean_list = Standard_DataFactors[:,RegionMask[0] > 0].mean(1, keepdims=True)
    mean_list = mean_list.reshape(mean_list.shape[0], mean_list.shape[1], 1)
    std_list = Standard_DataFactors[:, RegionMask[0] > 0].std(1, keepdims=True)
    std_list = std_list.reshape(std_list.shape[0], std_list.shape[1], 1)
    Norm_DataFactors = (Standard_DataFactors-mean_list)/std_list
# Norm_DataFactors = DataFactors
factors = [i for i in range(23)]
for i in rfac_index:
    Norm_DataFactors[i] = np.random.normal(0,1,(m,n))

Data4RF_dict = {'TPU': {
    'factor': [], 'pop': []}, 'SBG': {'factor': [], 'pop': []}}
Data4RF_dict['TPU']['factor'] = group_aggregation(Norm_DataFactors, RegionMask[1], False)[1][1:]
Data4RF_dict['SBG']['factor'] = group_aggregation(Norm_DataFactors, RegionMask[0], False)[1][1:]
Data4RF_dict['TPU']['pop'] = group_aggregation_(PopDensity[1], RegionMask[1], False)[1][1:]
Data4RF_dict['SBG']['pop'] = group_aggregation_(PopDensity[0], RegionMask[0], False)[1][1:]


in_data_grid = torch.from_numpy(
    Norm_DataFactors.reshape((c, m*n)).T).float().cuda()

in_data_aggr = torch.from_numpy(Data4RF_dict['TPU']['factor']).cuda().float()
                
Data4RF_dict['TPU']['pop'] = np.log10(Data4RF_dict['TPU']['pop'])

re_data_aggr_by_division =  torch.from_numpy(Data4RF_dict['TPU']['pop']).cuda().float().squeeze()

n_SBG = RegionMask[0].max()
n_TPU = RegionMask[1].max()


def validation(model):
    with torch.no_grad():
        data = in_data_grid
        pop_p = model(data).squeeze(-1).cpu().numpy().reshape((m, n))
        unit = 1
        population_potential_delog = np.power(10, pop_p)
        sum_potential_delog_by_org_untit = group_aggregation_(
            population_potential_delog, RegionMask[unit], keep_shape=True, method='sum')
        sum_population_by_org_untit = PopDensity[unit]*Area[unit]
        population_redistributed = population_potential_delog / \
            sum_potential_delog_by_org_untit * sum_population_by_org_untit
        # evaluation
        unit = 0
        population_redistributed_sum_by_su = group_aggregation_(
            population_redistributed, RegionMask[unit], False, method='sum')[1][1:]
        population_by_su_census_data = group_aggregation_(PopDensity[unit]*Area[unit], RegionMask[unit], False, method='mean')[1][1:]
        rmse = np.sqrt(metrics.mean_squared_error(
            population_by_su_census_data, population_redistributed_sum_by_su))
    return rmse

def evaluation(model):
    with torch.no_grad():
        model.eval()
        data = in_data_grid
        pop_p = model(data, train=True).squeeze(-1).cpu().numpy().reshape((m, n))
        unit = 1
        population_potential_delog = np.power(10, pop_p)
        sum_potential_delog_by_org_untit = group_aggregation_(
            population_potential_delog, RegionMask[unit], keep_shape=True, method='sum')
        sum_population_by_org_untit = PopDensity[unit]*Area[unit]
        population_redistributed = population_potential_delog / \
            sum_potential_delog_by_org_untit * sum_population_by_org_untit
        # evaluation
        unit = 0
        population_redistributed_sum_by_su = group_aggregation_(
            population_redistributed, RegionMask[unit], False, method='sum')[1][1:]
        population_by_su_census_data = group_aggregation_(PopDensity[unit]*Area[unit], RegionMask[unit], False, method='mean')[1][1:]
        rmse = np.sqrt(metrics.mean_squared_error(
            population_by_su_census_data, population_redistributed_sum_by_su))
        mae = metrics.mean_absolute_error(population_by_su_census_data, population_redistributed_sum_by_su)
        rmse_percent = rmse/population_by_su_census_data.mean()
        r2 = metrics.r2_score(population_by_su_census_data, population_redistributed_sum_by_su)
        with open('log/report.csv','a') as f:
            ckp_dis = ','.join(args.ckp.split('_'))
            print(f'{ckp_dis},{rmse},{mae},{rmse_percent},{r2}',file=f)
    return rmse

def aggragate_torch(value_array, regions,mode = 'mean'):
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
        vs[i] = aggr_method(value_array_sort[start:end]) # torch.mean(value_array_sort[start:end])
        # vs[i] = torch.sum(value_array_sort[start:end])
        start = end
    return vs.squeeze(-1)


EPOCH = args.ep
LR = args.lr
preset_flag = 0.5
CKP = args.ckp  
l2_loss = nn.MSELoss(reduction='none')
model = Model_SI_old(DataFactors.shape[0]).cuda()
weight = IndWeight(args.type,int(RegionMask[1].max())).cuda()
if args.mode=='train':
    optimizer = optim.Adam(list(model.parameters()) +
                        list(weight.parameters()), lr=LR)
if CKP != 'None':
    checkpoint = torch.load("F:/code/HKPop2021/model_para/"+CKP)
    model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
params_str = '_'.join([
args.ckp,
str(args.ep),
args.mode,
args.type,
'{:.2e}'.format(args.lr),
args.dis
])

def train():
    # 在结果中加上时间戳
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    model_flag = timestamp + '_' + params_str
    log_root = f"log/{model_flag}"
    print(log_root)
    summaryWriter = SummaryWriter(log_root)


    # summaryWriter = SummaryWriter(f"log/{FLAG}")
    loss_bar = tqdm(range(1, EPOCH+1))
    min_loss = 2000
    min_rmse = 1e7
    for epoch in loss_bar:
        if args.type == 'grid':
            pop_p_grid = model(in_data_grid).reshape(m, n)
            delog_pop_p_grid =  torch.pow(10, pop_p_grid)
            pop_p_grid_aggr_TPU = torch.log10(aggragate_torch(delog_pop_p_grid, torch.from_numpy(RegionMask[1]))[1:]).squeeze(-1)
            l_grid_tpu = l2_loss(pop_p_grid_aggr_TPU, re_data_aggr_by_division)
            loss_tpu = l_grid_tpu.mean()
        elif args.type == 'unit':
            pop_p_direct_aggr_TPU = model(in_data_aggr).squeeze(-1)
            l_unit_tpu = l2_loss(pop_p_direct_aggr_TPU, re_data_aggr_by_division)
            loss_tpu = l_unit_tpu.mean()
        else:
            pop_p_grid = model(in_data_grid).reshape(m, n)
            delog_pop_p_grid =  torch.pow(10, pop_p_grid)
            pop_p_grid_aggr_TPU = torch.log10(aggragate_torch(delog_pop_p_grid, torch.from_numpy(RegionMask[1]))[1:]).squeeze(-1)
            pop_p_direct_aggr_TPU = model(in_data_aggr).squeeze(-1)
            l_grid_tpu = l2_loss(pop_p_grid_aggr_TPU, re_data_aggr_by_division)
            l_unit_tpu = l2_loss(pop_p_direct_aggr_TPU, re_data_aggr_by_division)
            loss_tpu = weight(l_grid_tpu,l_unit_tpu) 


        loss = loss_tpu
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        rmse = validation(model)
        if min_rmse >= rmse.item():
            checkpoint = {'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'weight': weight.state_dict()}
            torch.save(checkpoint, f'model_para/{model_flag}.pth')
            min_rmse = rmse
        model.train()
        loss_bar.set_postfix({'loss:': '%.5f' % loss.item(
        ), 'RMSE:': '%.3f' % rmse, 'min_rmse:': '%.5f' % min_rmse})
        summaryWriter.add_scalar("loss", loss.item(), epoch)
        summaryWriter.add_scalar("rmse", rmse, epoch)


def copy_geoinfo_and_save_image(ref_img, array, save_path):

    ds = gdal.Open(ref_img)
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    ds = None

    metadata = {
        'count': 1,
        'height': array.shape[0],
        'width': array.shape[1],
        'dtype': str(array.dtype),
        'transform': geotransform,
        'crs': projection
    }
    print(metadata)
    array = np.array(array, dtype=metadata['dtype'])
    array = np.expand_dims(array, axis=0)

    if os.path.exists(save_path):
        raise ValueError('The save path already exists.')

    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(save_path, metadata['width'], metadata['height'], metadata['count'],gdal.GDT_Float32)
    ds.SetGeoTransform(metadata['transform'])
    ds.SetProjection(metadata['crs'])
    ds.GetRasterBand(1).WriteArray(array[0])
    ds.FlushCache()
    ds = None

def mapping():
    with torch.no_grad():
        data = in_data_grid
        pop_p = model(
            data, train=True).squeeze(-1).cpu().numpy().reshape((m, n))

        unit = 1
        population_potential_delog = np.power(10, pop_p)
        temp_mask = PopDensity[unit]

        temp_mask[RegionMask[0]==800] = 214

        sum_potential_delog_by_org_untit = group_aggregation_(
            population_potential_delog, temp_mask, keep_shape=True, method='sum')
        sum_population_by_org_untit = PopDensity[unit]*Area[unit]

        sum_population_by_org_untit[RegionMask[0]==800] = 1184

        population_redistributed = population_potential_delog / \
            sum_potential_delog_by_org_untit * sum_population_by_org_untit
        out_map = np.log10(population_redistributed/0.03/0.03)
        out_map[out_map==-np.inf] = 0
        population_redistributed_sum_by_su = group_aggregation_(
            population_redistributed, RegionMask[0], False, method='sum')[1][1:]
        np.savetxt(f'map/{args.ckp}.csv',population_redistributed_sum_by_su,delimiter=',')
        copy_geoinfo_and_save_image(args.map, out_map, args.savepath)
def get_feature():
    model.eval()
    print('Get feature')
    with torch.no_grad():
        f_grid = model(in_data_grid,train=False).reshape(32,m, n)
        f_grid = f_grid.cpu().detach().numpy()
        np.save(f'log/feature_{args.year}_{args.type}.npy',f_grid)
        
if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'map':
        mapping()     
    elif args.mode=='feature':
        get_feature()  
    else:
        evaluation(model)
