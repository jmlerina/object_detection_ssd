### Real-time Word Detection

```bash
#Prepare the model
mkdir models; cd models
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar xzvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz; rm ssd_mobilenet_v2_coco_2018_03_29.tar.gz
mv coco_labels.txt models/ssd_mobilenet_v2_coco_2018_03_29/

#Setup conda
conda env create -f windows_env.yml|linux_env.yml
activate object_detection

# ==> Check model and labels paths within the script
#Run real-time inferences
python od_ssd_local.py
```

