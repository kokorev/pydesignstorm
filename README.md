# pydesignstorm
Creating Design Storm for openLISEM

## Work Flow
Download ERA5 precipitation data for the region using *download.py*

Create a bias corrected version of ERA5 data by adjusting a distribution to fit the observed one (*bias_correction.py*)

Create and Export design storm scenarios (designers.py; basic example in example.ipynb)

## Dependencies
following Python pakages are requaried 
- pandas
- numpy
- netCDF4
- matplotlib
- scipy
- pyshp
- shapely
- cmocean
- pillow (for writing tif files)
- jupyter (only for example nootbooks)

## Enviroment install example
```bash
conda create -n ENV_NAME pandas numpy netCDF4 matplotlib scipy pyshp shapely pillow
conda activate ENV_NAME
pip install cmocean
```
