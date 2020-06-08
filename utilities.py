import tensorflow as tf
import tensorflow_hub as hub
import json


image_size = 224

def process_image(npimage):
    tensor = tf.image.convert_image_dtype(npimage, dtype=tf.int16, saturate=False)
    resizeimage = (tf.image.resize(npimage,(image_size,image_size)).numpy())/255
    return resizeimage


def label_mapping(json_file):
    with open(json_file, 'r') as f:
    	class_names = json.load(f)
    return class_names
