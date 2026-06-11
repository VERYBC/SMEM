#  <div align="center">Sparse Local Feature Matching Enhancement Method for UAV Cross-view Absolute Localization</div>
This repository contains the dataset and the code for our paper Sparse Local Feature Matching Enhancement Method for UAV Cross-view Absolute Localization. Thank you for your kindly attention.
## 1. Test on XIAN-Visloc
### 1.1. Data preprocess  

1. Download the [XIAN-Visloc data](https://huggingface.co/datasets/VERYBC/XIAN_Visloc) and place it in the `data/` folder.

2. Construct the satellite reference library:
   - Run `scripts/XIAN_Visloc_satellite.py `. Set the satellite image resolution to **(392, 392)**.

3. Resize UAV images:
   - Run `scripts/UAV image rescaling.py`. The processed UAV data will be saved to the `drone_392/` folder.
  
### 1.2. Download LightGlue weights 

- Download the model weight files [[Google]](https://drive.google.com/drive/folders/1ug7TvR-SMjKRQ22qfhCPBLS4tNJtRTmU?usp=drive_link) and place them in the `checkpoints/` directory.
  
### 1.3. Localization
- Run `python location.py`, you will get the localization results, and the results will be saved to the `result/` folder.
