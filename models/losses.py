import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss

class WBCE(Loss):
    def __init__(self, weights=1.0, class_indexes = None, **kwargs):
        super(WBCE, self).__init__(**kwargs)
        self.weights = weights
        #self.weights = tf.constant([weights])
        self.class_indexes = class_indexes

    def __call__(self, y_true, y_pred, **kwargs): 
        return self.call(y_true, y_pred, **kwargs)
    
    def call(self, y_true, y_pred, **kwargs):
        #opt_y_pred = y_pred
        #sar_y_pred = y_pred[1]
        #fusion_y_pred = y_pred[2]
        weights = self.weights
        
        #filter the classes indexes       
        if self.class_indexes is not None:
            y_true = tf.gather(y_true, self.class_indexes, axis=3)
            y_pred = tf.gather(y_pred, self.class_indexes, axis=3)
            weights = tf.gather(weights, self.class_indexes, axis=0)


        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        loss = y_true * tf.math.log(y_pred) + (1-y_true) * tf.math.log(1-y_pred)
        loss = loss * weights 
        loss = - tf.math.reduce_mean(loss, -1)
        #keep = tf.math.argmax(y_true, axis=-1) != 2
        #loss = tf.reshape(loss, [-1])[tf.reshape(keep, [-1])]
        return tf.math.reduce_mean(loss)
