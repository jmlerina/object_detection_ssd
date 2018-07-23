# Real-time Object Detection using SSD Mobilenet

Object detection that uses OpenCV and Tensorflow to execute in real-time.
Model was obtained from here (http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz).
Other models available are listed here (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

```bash
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
