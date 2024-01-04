import numpy as np 
import tensorflow as tf 
import os 

raw_image_dir = 'images'
target_numpy_dir = 'numpy-images'

def process(image_file, out_dir):
    img = tf.keras.preprocessing.image.load_img(image_file, target_size=(224,224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis =0)
    x = tf.keras.applications.vgg16.preprocess_input(x)
    print(x.shape, x.dtype)
    
    bname, ext = os.path.splitext(image_file)
    fname = os.path.basename(bname)
    fname = fname+'.npy'
    abs_fname = os.path.join(out_dir, fname)
    
    np.save(abs_fname, x)
    
if __name__=="__main__":
    current_dir = os.path.abspath(os.getcwd())
    input_dir = os.path.join(current_dir, raw_image_dir)
    output_dir = os.path.join(current_dir, target_numpy_dir)
    
    test_pics = os.listdir(input_dir)
    for tp in test_pics:
        full_path = os.path.join(input_dir, tp)
        process(full_path, output_dir)