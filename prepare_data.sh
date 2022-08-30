echo "run the 0_MergeImages.py"
python src/classification/etl/0_MergeImages.py

echo "run the 1_MergeChexpert.py"
python src/classification/etl/1_MergeChexpert.py

echo "run the 2_MergeDataSet.py"
python src/classification/etl/2_MergeDataSet.py

echo "run the 3_AssignDataSet.py"
python src/classification/etl/3_AssignDataSet.py

echo "run the 4_ApplySegmentation.py"
python src/classification/etl/4_ApplySegmentation.py

echo "run the 5_AssignModelDataSet.py"
python src/classification/etl/5_AssignModelDataSet.py

echo "run the 6_AssignTrain_ValSet.py"
python src/classification/etl/6_AssignTrain_ValSet.py