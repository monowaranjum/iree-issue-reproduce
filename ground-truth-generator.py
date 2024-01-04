import numpy as np
import tensorflow as tf 
import os

model = tf.saved_model.load('./saved-model-signed')

a, c = model.custom_predict(np.zeros((1,224,224,3)))
b, d = model.custom_predict(np.ones((1,224,224,3)))

print('all zeros: ', a.numpy(), c.numpy())
print('all ones: ', b.numpy(), d.numpy())

numpy_img_list = os.listdir('./numpy-images')

for img in numpy_img_list:
    testcase = np.load('./numpy-images/{}'.format(img))
    result = model.custom_predict(testcase)
    print(img, result[0].numpy(), result[1].numpy())
    
print("Ground truth generation complete")





