import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from utilities import process_image, label_mapping
import json
import numpy as np
import argparse


def predict(image_path, saved_model, top_k, category_names):
       
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    processed_test_image = np.expand_dims(processed_test_image, axis=0)    
    load_model = tf.keras.models.load_model(saved_model,custom_objects={'KerasLayer': hub.KerasLayer})
    ps = load_model.predict(processed_test_image)       
    
    # top flower prediction
    val,ind = tf.math.top_k(ps[0], k=1)
    probs= list(val.numpy())
    classes= list(ind.numpy())
    print('top prediction class is :\n',classes)
    print('top prediction label is :\n',[category_names[str(n+1)] for n in classes])
    print('probability of it is :\n',probs)
    
    # top k flower predictions
    val,ind = tf.math.top_k(ps[0], k=top_k)
    probs= list(val.numpy())
    classes= list(ind.numpy())
    print('classes of top k are:\n',classes)
    print('labels of top k are:\n',[category_names[str(n+1)] for n in classes])
    print('probabilities of top k are :\n',probs)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Processing flower prediction')
    parser.add_argument('image_path',default='test_images/wild_pansy.jpg', type = str)
    parser.add_argument('saved_model',type= str)
    parser.add_argument('--top_k', required = False, default = 5,type = int)
    parser.add_argument('--category_names', required = False, default = 'label_map.json', type = str)
    args = parser.parse_args()    
    category_names = label_mapping(args.category_names)
    predict(args.image_path,args.saved_model,args.top_k,category_names)