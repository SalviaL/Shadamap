# SHAF4DM
An ANN-based Framework for Population Dasymetric Mapping to Avoid the Scale Heterogeneity

## Dataset
This project provides two geospatial dataset in Hong Kong in 2016 and 2021. You can download it by following [Dataset link][https://drive.google.com/drive/folders/1-HIdq1tPI3eqSXbCDcV-qN29adj8fgjg?usp=sharing]. 
These two datasets are composited tif file (30 meter, Hong Kong 1980 Grid coordination system) with 28 bands (layers), the meaning fo each band is listed in the following tabel:
|Band #|Meaning|
|---|---|
|0-3|NIR and RGB bands from Landsat|
|4|NDVI|
|5|Road kernel density|
|6|Closest distance to road|
|7|DTM|
|8|Slope|
|9|Building fraction|
|10|Average building height|
|11-22|Land use fraction: residential land, commercial land, industrial land, public land, transportation, airport,, unclassified, harbor, agricultural field, vegetation, and water|
|23-25|ID, area(km2), and population density (/km2) by TPSU|
|26-28|ID, area(km2), and population density (/km2) by TPU|

## Methods
Please refer to our paper (`TODO`) and the code (1main.py1 and 1models.py1). The core part is in lines 276-278:
'''python
pop_p_grid = model(in_data_grid).reshape(m, n)
delog_pop_p_grid =  torch.pow(10, pop_p_grid)
pop_p_grid_aggr_TPU = torch.log10(aggragate_torch(delog_pop_p_grid, torch.from_numpy(RegionMask[1]))[1:]).squeeze(-1)
l_grid_tpu = l2_loss(pop_p_grid_aggr_TPU, re_data_aggr_by_division)
loss_tpu = l_grid_tpu.mean()
'''
