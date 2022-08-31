# Chest-Xray-Classification
   
<p align="left"><img src="https://user-images.githubusercontent.com/63839581/118935419-7e42df00-b986-11eb-911f-80173174977d.jpg"></p>
    
## Description

This repository aims to classify six disease types, including Consolidation, Pneumothorax, Edema, Effusion, Pneumonia, and Cardiomegaly, present in Chest X-Ray images through CNN.

The classification data set was utilized by merging **NIH Datset** and **Chexpert Dataset**, and the lung location segmentation data set was self-made. The project largely performed **preprocessing, learning, visualization (GradCam) and detection, and evaluation**.   
   
## Requirement
   

Requires Python 3.6 and keras 2.2.4, tensorflow-gpu 1.14.0.
Build your environment via the provided [requirements.txt](https://github.com/sckim0430/Chest-Xray-Classification/blob/master/requirements.txt).
    
```bash
$ pip install -r requirements.txt
```   
   
## Custom Dataset   

1. Modify [SETTINGSjson](https://github.com/sckim0430/Chest-Xray-Classification/blob/master/SETTINGS.json) that specifies the path of files and folders and [PARAMS.json](https://github.com/sckim0430/Chest-Xray-Classification/blob/master/PARAMS.json) that organizes parameters.

2. Modify [PARAMS.json](https://github.com/sckim0430/Chest-Xray-Classification/blob/master/PARAMS.json) in which the paths of files and folders are specified and parameters are arranged.
   
3. [NIH DataSet](https://www.kaggle.com/nih-chest-xrays/data) and [Chexpert Dataset](https://stanfordmlgroup.github.io/competitions/chexpert/) are saved in the ./data/classification folder, and the self-made data set is saved in the ./data/segmentation/dataset folder.
   
3. Run [prepare_data.sh](https://github.com/sckim0430/Chest-Xray-Classification/blob/master/prepare_data.sh) file
```bash
$ sh prepare_data.sh
```   
   
## Train / Evaluation / Visualization
   
The Train course is largely divided into **Segmentation** and **Classification Module**.

The Segmentation Module uses **UNet Model**, which is mainly used in the medical image processing field.
   
1. Start training by executing [train.py](https://github.com/sckim0430/Chest-Xray-Classification/blob/master/src/segmentation/train.py) that exists in the ./src/segmentation folder.
   
```bash
$ python src/segmentation/train.py
```   
   
2. The trained UNet model is stored in models/segmentation/Unet/weights/ location, and run [evaluate.py](https://github.com/sckim0430/Chest-Xray-Classification/blob/master/src/segmentation/evaluate.py) to perform model performance evaluation.
   
```bash
$ python src/segmentation/evaluate.py
```   
    
3. Check the inference result image of the model through [predict.py](https://github.com/sckim0430/Chest-Xray-Classification/blob/master/src/segmentation/predict.py) and repeat steps 1 and 2 above to improve performance.
   
```bash
$ python src/segmentation/predict.py
```   
   
<p align="left"><img src="https://user-images.githubusercontent.com/63839581/118943991-1d6bd480-b98f-11eb-947d-488f6881846c.jpg"></p>
   
Classification Module used **Ensemble Model**(densenet169, inceptionV3, Xception) and **EfficientNet**.

When learning **DenseNet Model**
   
1. Run **train_DenseNet.sh** to proceed with training.
   
```bash
$ sh train_DenseNet.sh
```   
  
2. Run **evaluate_Ensemble.sh** to evaluate the trained model.
   
```bash
$ sh evaluate_Ensemble.sh
```   
  
3. Execute **visualization_Ensemble.sh** to check the inference result image of the model and repeat steps 1 and 2 above to improve performance.
   
```bash
$ sh visualization_Ensemble.sh
```   
  
When training **EfficientNet Model**
   
1. Run **train_Efficientnet.sh** to proceed with training.
   
```bash
$ sh train_Efficientnet.sh
```   
  
2. Evaluate the trained model by running **evaluate_Efficientnet.sh**.
   
```bash
$ sh evaluate_Efficientnet.sh
```   
  
3. Execute **visualization_Efficientnet.sh** to check the inference result image of the model and repeat steps 1 and 2 above to improve performance.
   
```bash
$ sh visualization_Efficientnet.sh
```   
   
<p align="left"><img src="https://user-images.githubusercontent.com/63839581/118946998-e0eda800-b991-11eb-8b91-0f99219ccc8e.jpg"></p>
   
## Result
|disease|Accuracy|Precision|Recall|
|:------:|:---:|:---:|:---:|
|Consolidation|0.9670|0.8474|0.8695|
|Pneumothorax|0.9710|0.8791|0.9225|
|Edema|0.8870|0.7777|0.9102|
|Effusion|0.8840|0.9214|0.8814|
|Pneumonia|0.9830|0.9285|0.6341|
|Cardiomegaly|0.9350|0.8514|0.7925|
|Total|0.9378|0.8588|0.8810|

## Contact
   
another0430@naver.com
  
