## Real-time Detection

See the original project for more info.
Mind the path strings in the code and prefer using absolute paths.

### Object detection :: MobileNet
```bash
cd object_detection
#Prepare the model
mkdir models
cd models
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar xzvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz; rm ssd_mobilenet_v2_coco_2018_03_29.tar.gz
cd ..
mv coco_labels.txt models/ssd_mobilenet_v2_coco_2018_03_29/

#Setup conda
conda env create -f linux_env.yml
source activate od

# ==> Check model and labels paths within the script
#Run real-time inferences
python od_ssd_local.py
```
