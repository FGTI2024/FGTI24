# Frequency-aware Generative Models for Multivariate Time Series Imputation


## File Structure
* Code: source code of our implementation
* Data: some source files of datasets used in experiments


## Preprocessing each dataset
0. Enter the "Code" folder

1. To get KDD dataset:
```
python loadKddDataset.py
```

2. To get Guangzhou dataset:
```
python loadGuangzhouDataset.py
```
3. To get PhysioNet dataset:
```
python loadPhysioDataset.py
```

## Demo Script Running
```
python A_diffusion_train.py --dataset kdd --missing_rate 0.1 --enc_in 99 --c_out 99
```

```
python A_diffusion_train.py --dataset guangzhou --missing_rate 0.1 --enc_in 214 --c_out 214
```

```
python A_diffusion_train.py --dataset physio  --missing_rate 0.1 --enc_in 37 --c_out 37
```

## Dataset Sources
* KDD: http://www.kdd.org/kdd2018/
* Guangzhou: https://zenodo.org/record/1205229
* PhysioNet: https://physionet.org/content/challenge-2012/1.0.0/
