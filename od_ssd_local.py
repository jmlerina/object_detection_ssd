#!/usr/bin/env python
import cv2
import sys
import json
import numpy as np
import tensorflow as tf

camera = cv2.VideoCapture(0)

def grabVideoFeed():
    grabbed, frame = camera.read()
    return frame if grabbed else None

def load_coco_dict(labels_path):
    with open(labels_path, 'r') as f:
        coco_labels = json.load(f)

    labels_dict = {}
    for info in coco_labels['labels']:
        labels_dict[info['id']] = info['label']

    return labels_dict


class image_converter:

  def __init__(self):
    print ("Instatiating Net...")
    model_checkpoint = '/models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'
    labels_path = '/models/ssd_mobilenet_v2_coco_2018_03_29/coco_labels.txt'
    self.labels_dict = load_coco_dict(labels_path)

    #Load the model graph from file and create a session.
    #  Here, we create a session that will be kept in memory and reused for all inferences.
    self.detection_graph = tf.Graph()
    self.session = tf.Session(graph=self.detection_graph)
    with self.detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_checkpoint, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        with self.session as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')


  ##########################
  #Method to run inference using the graph defined at 'init'
  def run_inference(self, image, graph):
    with graph.as_default():
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
          tensor_name = key + ':0'
          if tensor_name in all_tensor_names:
              tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = self.session.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
    return output_dict


  ##########################
  #Callback in the event of receiving new frame from camera topic
  def callback(self, cv_image):
    try:
      cv_image_prep = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
      (height, width, channels) = cv_image.shape
    except CvBridgeError as e:
      print(e)
      raise

    output_dict = self.run_inference(cv_image_prep, self.detection_graph)

    confidence_threshold = .6

    for boxes, score, cls_id in zip(output_dict['detection_boxes'], output_dict['detection_scores'], output_dict['detection_classes']):
        ymin, xmin, ymax, xmax = boxes
        if score > confidence_threshold:
            (left, right, top, bottom) = (xmin * width, xmax * width,
                                          ymin * height, ymax * height)
            cv2.rectangle(cv_image, (left.astype("int"),top.astype("int")), (right.astype("int"),bottom.astype("int")), (0,0,255), 2)
            print(self.labels_dict[cls_id])
    cv2.imshow("Test", cv_image)
    cv2.waitKey(1)


def main(args):
  ic = image_converter()

  while True:
    frame = grabVideoFeed()

    if frame is None:
        raise SystemError('Issue grabbing the frame')

    ic.callback(frame)
  cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
