# SHAF4DM

*Please note that this project is currently under construction.*

## Dataset

This project provides two geospatial datasets for Hong Kong in 2016 and 2021. You can download them by following this [Dataset link](https://drive.google.com/drive/folders/1-HIdq1tPI3eqSXbCDcV-qN29adj8fgjg?usp=sharing).

These two datasets are composed of TIFF files (30 meters, Hong Kong 1980 Grid coordination system) with 28 bands (layers). The meaning of each band is listed in the following table:

| Band # | Meaning |
| --- | --- |
| 0-3 | NIR and RGB bands from Landsat |
| 4 | NDVI |
| 5 | Road kernel density |
| 6 | Closest distance to road |
| 7 | DTM |
| 8 | Slope |
| 9 | Building fraction |
| 10 | Average building height |
| 11-22 | Land use fraction: residential land, commercial land, industrial land, public land, transportation, airport, unclassified, harbor, agricultural field, vegetation, and water |
| 23-25 | ID, area (km2), and population density (/km2) by TPSU |
| 26-28 | ID, area (km2), and population density (/km2) by TPU |

## Methods

Please refer to our paper (`TODO`) and the code (`main.py` and `models.py`). The core part of the method is shown below:

```python
pop_p_grid = model(in_data_grid).reshape(m, n)
delog_pop_p_grid = torch.pow(10, pop_p_grid)
pop_p_grid_aggr_TPU = torch.log10(aggragate_torch(delog_pop_p_grid, torch.from_numpy(RegionMask[1]))[1:]).squeeze(-1)
l_grid_tpu = l2_loss(pop_p_grid_aggr_TPU, re_data_aggr_by_division)
loss_tpu = l_grid_tpu.mean()
```

Line 148-153 of the code shows how the population prediction grid is generated using an ANN-based framework. The code uses PyTorch to train the model and predict population density in different areas. The paper provides more detailed information about the methodology used.

## Result
![](PopDensity.png)
We provide two verified population distribution maps (in 2016 and 2021) and an unverified map (in 2019) with a 30-meter resolution. The verified maps are generated by models trained on corresponding temporal data. We use the census population by Tertiary Planning Units (TPU) as the redistribution source unit and the Tertiary Planning Subunits (TPSU) as the evaluation unit. The map in 2019 is the average of the results of the models trained on data from 2016 and 2021, and predicted on data from 2019. The following table shows the quantitative indices of the maps in 2016 and 2021.

| Year | RMSE    | MAE     | $R^2$     | %RMSE   |
|------|---------|---------|---------|---------|
| 2016 | 1528.25 | 802.95  | 0.90    | 47.9    |
| 2021 | 1464.14 | 786.06  | 0.91    | 46.3    |







