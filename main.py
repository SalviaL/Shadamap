import time
from models import *
from osgeo import gdal
import numpy as np
from sklearn import metrics
import os
import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from tools import *
# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='这是程序的描述信息')

# 添加参数
parser.add_argument('--ckp', type=str, default='None', help='模型文件的路径')
parser.add_argument('--ep', type=int, default=50000, help='训练的轮数')
parser.add_argument('--mode', type=str,
                    choices=['train', 'map', 'eval'], default='train', help='程序的模式')
parser.add_argument('--type', type=str, default='both', help='损失函数')
parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
parser.add_argument('--dis', '-d', type=str, default='', help='额外描述')
parser.add_argument('--year', '-y', type=str, default='2016', help='数据年份')
parser.add_argument('--savepath', '-s', type=str, default='None')
# 解析参数
args = parser.parse_args()
args.map = "data_lib/" + \
    {'2021': "HongKong2021_FeatureCube.tif",
        '2016': 'HongKong2016_FeatureCube.tif'}[args.year]

if (args.mode == 'map') & (args.savepath == 'None'):
    raise ValueError('Please set the save path.')
# 使用参数
print("ckp 文件路径：", args.ckp, end='')
print("训练轮数：", args.ep, end='')
print("程序模式：", args.mode, end='')
print("数据类型：", args.type, end='')
print("学习率：", args.lr, end='')
print("额外描述：", args.dis, end='')
print("年份：", args.year, end='')


# load and split data HK
Dataset = gdal.Open(
    args.map).ReadAsArray()

# factors = tuple(factors)
DataFactors = Dataset[:-6]
DataFactors[DataFactors > 1e30] = 0
DataFactors[DataFactors <= -1e38] = 0

PopDensity = Dataset[(-4, -1), :, :]
Area = Dataset[(-5, -2), :, :]
RegionMask = Dataset[(-6, -3), :, :]
c, m, n = DataFactors.shape

'''For data in 2016, please add the following 3 patches'''
'''For data in 2016, please add the following 3 patches'''
'''For data in 2016, please add the following 3 patches'''
# RegionMask[1][RegionMask[0] == 800] = 214
# PopDensity[1][RegionMask[0] == 800] = 25.806874
# Area[1][Area[0] == 800] = 45.879248

max_list = DataFactors[:, RegionMask[0] > 0].max(1, keepdims=True)
max_list = max_list.reshape(max_list.shape[0], max_list.shape[1], 1)
min_list = DataFactors[:, RegionMask[0] > 0].min(1, keepdims=True)
min_list = min_list.reshape(min_list.shape[0], min_list.shape[1], 1)
Standard_DataFactors = (DataFactors - min_list)/(max_list-min_list)
mean_list = Standard_DataFactors[:, RegionMask[0] > 0].mean(1, keepdims=True)
mean_list = mean_list.reshape(mean_list.shape[0], mean_list.shape[1], 1)
std_list = Standard_DataFactors[:, RegionMask[0] > 0].std(1, keepdims=True)
std_list = std_list.reshape(std_list.shape[0], std_list.shape[1], 1)
Norm_DataFactors = (Standard_DataFactors-mean_list)/std_list

Group_TPU = {}
Group_TPU['factor'] = group_aggregation(
    Norm_DataFactors, RegionMask[1], False)[1][1:]
Group_TPU['pop'] = group_aggregation_(
    PopDensity[1], RegionMask[1], False)[1][1:]


in_data_grid = torch.from_numpy(
    Norm_DataFactors.reshape((c, m*n)).T).float().cuda()

in_data_aggr = torch.from_numpy(Group_TPU['factor']).cuda().float()

Group_TPU['pop'] = np.log10(Group_TPU['pop'])

re_data_aggr_by_division = torch.from_numpy(
    Group_TPU['pop']).cuda().float().squeeze()

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
        population_by_su_census_data = group_aggregation_(
            PopDensity[unit]*Area[unit], RegionMask[unit], False, method='mean')[1][1:]
        rmse = np.sqrt(metrics.mean_squared_error(
            population_by_su_census_data, population_redistributed_sum_by_su))
    return rmse


EPOCH = args.ep
LR = args.lr
CKP = args.ckp
l2_loss = nn.MSELoss(reduction='none')
model = Model_SI(DataFactors.shape[0]).cuda()

if args.mode == 'train':
    optimizer = optim.Adam(model.parameters())
if CKP != 'None':
    checkpoint = torch.load("checkpoints/"+CKP)
    model.load_state_dict(checkpoint['model'])
params_str = '_'.join([
    args.ckp,
    str(args.ep),
    args.mode,
    args.type,
    '{:.2e}'.format(args.lr),
    args.dis
])


def train():
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    model_flag = timestamp + '_' + params_str
    log_root = f"log/{model_flag}"
    print(log_root)
    summaryWriter = SummaryWriter(log_root)
    loss_bar = tqdm(range(1, EPOCH+1))
    min_loss = 2000
    min_rmse = 1e7
    for epoch in loss_bar:
        pop_p_grid = model(in_data_grid).reshape(m, n)
        delog_pop_p_grid = torch.pow(10, pop_p_grid)
        pop_p_grid_aggr_TPU = torch.log10(aggragate_torch(
            delog_pop_p_grid, torch.from_numpy(RegionMask[1]))[1:]).squeeze(-1)
        l_grid_tpu = l2_loss(pop_p_grid_aggr_TPU, re_data_aggr_by_division)
        loss_tpu = l_grid_tpu.mean()

        loss = loss_tpu
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        rmse = validation(model)
        if min_rmse >= rmse.item():
            checkpoint = {'model': model.state_dict(),
                          'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, f'checkpoints/{model_flag}.pth')
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
    array = np.array(array, dtype=metadata['dtype'])
    array = np.expand_dims(array, axis=0)

    if os.path.exists(save_path):
        raise ValueError('The save path already exists.')

    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(
        save_path, metadata['width'], metadata['height'], metadata['count'], gdal.GDT_Float32)
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

        sum_potential_delog_by_org_untit = group_aggregation_(
            population_potential_delog, RegionMask[unit], keep_shape=True, method='sum')
        sum_population_by_org_untit = PopDensity[unit]*Area[unit]

        population_redistributed = population_potential_delog / \
            sum_potential_delog_by_org_untit * sum_population_by_org_untit
        out_map = np.log10(population_redistributed/0.03/0.03)
        out_map[out_map == -np.inf] = 0
        population_redistributed_sum_by_su = group_aggregation_(
            population_redistributed, RegionMask[0], False, method='sum')[1][1:]
        np.savetxt(f'map/{args.ckp}.csv',
                   population_redistributed_sum_by_su, delimiter=',')
        copy_geoinfo_and_save_image(args.map, out_map, args.savepath)


def evaluation(model):
    with torch.no_grad():
        model.eval()
        data = in_data_grid
        pop_p = model(
            data, train=True).squeeze(-1).cpu().numpy().reshape((m, n))
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
        population_by_su_census_data = group_aggregation_(
            PopDensity[unit]*Area[unit], RegionMask[unit], False, method='mean')[1][1:]
        rmse = np.sqrt(metrics.mean_squared_error(
            population_by_su_census_data, population_redistributed_sum_by_su))
        mae = metrics.mean_absolute_error(
            population_by_su_census_data, population_redistributed_sum_by_su)
        rmse_percent = rmse/population_by_su_census_data.mean()
        r2 = metrics.r2_score(population_by_su_census_data,
                              population_redistributed_sum_by_su)
        ckp_dis = ','.join(args.ckp.split('_'))
        print(f'{ckp_dis},{rmse},{mae},{rmse_percent},{r2}')


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'map':
        mapping()
    else:
        evaluation(model)
