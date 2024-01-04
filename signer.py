import tensorflow as tf

class TestModule(tf.Module):
    def __init__(self, unsigned_model_path):
        super(TestModule, self).__init__()
        self.unsigned_model = tf.keras.models.load_model(unsigned_model_path)
    
    @tf.function(
        input_signature=[
            tf.TensorSpec([1,224,224,3], tf.float32)
        ]
    )    
    def custom_predict(self, x):
        out = self.unsigned_model(x)
        return out, tf.nn.softmax(out)
    
if __name__=='__main__':
    tm = TestModule('./saved-model-unsigned')
    tf.saved_model.save(tm, './saved-model-signed')