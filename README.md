# Chest-Xray-Classification
   
<p align="left"><img src="https://user-images.githubusercontent.com/63839581/118935419-7e42df00-b986-11eb-911f-80173174977d.jpg"></p>
    
## Description
   
이 저장소는 CNN을 통해 Chest X-Ray 이미지에 존재하는 **Consolidation, Pneumothorax, Edema, Effusion, Pneumonia, Cardiomegaly**을 포함한 6가지 병종을 분류하는 것을 목표로 합니다.   
   
분류 데이터 셋은 **NIH Datset**과 **Chexpert Dataset**을 병합하여 활용했으며, 폐 위치 분할 데이터 셋은 자체 제작했습니다. 프로젝트는 크게 **전처리, 학습, 시각화(GradCam) 및 검출 그리고 평가 과정**을 진행했습니다.   
   
## Requirement
   
Python 3.6 및 keras 2.2.4, tensorflow-gpu 1.14.0이 필요합니다. 제공된 [requirements.txt](https://github.com/sckim0430/Chest-Xray-Classification/blob/master/requirements.txt)을 통해 환경을 구축하세요.     
    
```bash
$ pip install -r requirements.txt
```   
   
## Custom Dataset   
   
1. 파일과 폴더의 경로를 지정한 [SETTINGSjson](https://github.com/sckim0430/Chest-Xray-Classification/blob/master/SETTINGS.json)과 파라미터들을 정리한 [PARAMS.json](https://github.com/sckim0430/Chest-Xray-Classification/blob/master/PARAMS.json)을 수정합니다.   
   
2. ./data/classification 폴더에는 [NIH DataSet](https://www.kaggle.com/nih-chest-xrays/data)과 [Chexpert Dataset](https://stanfordmlgroup.github.io/competitions/chexpert/)을 저장하고, ./data/segmentation/dataset 폴더에는 자체 제작한 데이터 셋을 저장합니다.
   
3. [prepare_data.sh](https://github.com/sckim0430/Chest-Xray-Classification/blob/master/prepare_data.sh)을 실행합니다.   
   
```bash
$ sh prepare_data.sh
```   
   
## Train / Evaluation / Visualization
   
Train 과정은 크게 **Segemtation, Classification Module**로 구분됩니다.   
   
Segmentation Module은 의료 영상처리 분야에서 주로 사용되는 **UNet Model**을 사용했습니다.   
   
1. ./src/segmentation 폴더에 존재하는 [train.py](https://github.com/sckim0430/Chest-Xray-Classification/blob/master/src/segmentation/train.py)을 실행하여 학습을 시작합니다.   
   
```bash
$ python src/segmentation/train.py
```   
   
2. 학습된 UNet 모델은 models/segmentation/Unet/weights/ 위치에 저장되며, [evaluate.py](https://github.com/sckim0430/Chest-Xray-Classification/blob/master/src/segmentation/evaluate.py)을 실행하여 모델 성능 평가를 수행합니다.   
   
```bash
$ python src/segmentation/evaluate.py
```   
    
3. [predict.py](https://github.com/sckim0430/Chest-Xray-Classification/blob/master/src/segmentation/predict.py)을 통해 모델의 추론 결과 영상을 확인하고 위의 1,2 과정을 반복하여 성능을 개선시킵니다.   
   
```bash
$ python src/segmentation/predict.py
```   
   
<p align="left"><img src="https://user-images.githubusercontent.com/63839581/118943991-1d6bd480-b98f-11eb-947d-488f6881846c.jpg"></p>
   
Classification Module은 **Ensemble Model**(densenet169, inceptionV3, Xception)과 **EfficientNet**을 사용했습니다.

**Ensemble Model**을 학습하는 경우
   
1. **train_Ensemble.sh**를 실행하여 학습을 진행합니다.
   
```bash
$ sh train_Ensemble.sh
```   
  
2. **evaluate_Ensemble.sh**를 실행하여 학습된 모델을 평가합니다.
   
```bash
$ sh evaluate_Ensemble.sh
```   
  
3. **visualization_Ensemble.sh**를 실행하여 모델의 추론 결과 영상을 확인하고 위의 1,2 과정을 반복하여 성능을 개선시킵니다.
   
```bash
$ sh visualization_Ensemble.sh
```   
  
**EfficientNet Model**을 학습하는 경우
   
1. **train_Efficientnet.sh**를 실행하여 학습을 진행합니다.
   
```bash
$ sh train_Efficientnet.sh
```   
  
2. **evaluate_Efficientnet.sh**를 실행하여 학습된 모델을 평가합니다.
   
```bash
$ sh evaluate_Efficientnet.sh
```   
  
3. **visualization_Efficientnet.sh**를 실행하여 모델의 추론 결과 영상을 확인하고 위의 1,2 과정을 반복하여 성능을 개선시킵니다.
   
```bash
$ sh visualization_Efficientnet.sh
```   
   
<p align="left"><img src="https://user-images.githubusercontent.com/63839581/118946998-e0eda800-b991-11eb-8b91-0f99219ccc8e.jpg"></p>
   
## Result

|disease|Accuracy|Precision|Recall|
|------|---|---|---|
|Consolidation|0.9670|0.8474|0.8695|
|Pneumothorax|0.9710|0.8791|0.9225|
|Edema|0.8870|0.7777|0.9102|
|Effusion|0.8840|0.9214|0.8814|
|Pneumonia|0.9830|0.9285|0.6341|
|Cardiomegaly|0.9350|0.8514|0.7925|
|Total|0.9378|0.8588|0.8810|


## Contact
   
another0430@naver.com
  
