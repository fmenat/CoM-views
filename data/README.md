# Dataset creation

In order to create the dataset structures we used, you can execute the following code. 


### CropHarvest (binary and multi)  
```
python data_cropharvest.py -d DIRECTORY_RAW_DATA -o OUTPUT_DIR -c CROP_CASE
```
* options for CROP_CASE: [binary, multi]

> [!IMPORTANT]  
> The original data comes from [LFMC from SAR](https://github.com/kkraoj/lfmc_from_sar)

### Live Fuel Moisture Content  
```
python data_lfmc.py -d DIRECTORY_RAW_DATA -o OUTPUT_DIR 
```

> [!IMPORTANT]
> The original data comes from [LFMC from SAR](https://github.com/kkraoj/lfmc_from_sar)

### PM25 in Five Chinese Cities 
```
python data_pm25.py -d DIRECTORY_RAW_DATA -o OUTPUT_DIR -D 3
```
* D is the number of days to consider as input before the pm25 measurement. Common options are 3, 5, and 7.


> [!IMPORTANT]  
> The original data comes from [PM25 in UCI](https://doi.org/10.24432/C52K58)

 
---


> [!TIP] 
> Preprocessed data can be accessed at: [Link](https://cloud.dfki.de/owncloud/index.php/s/yxAfArTXkMF7nM2)
