import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

class tf_FC(object):

    def __init__(self):
        
        self._lr = 0.001
        self._batch_size = 512
        self._num_epoch = 5
        
        self._loss_train = np.zeros([(10000 // self._batch_size) * self._num_epoch , 1])

    """
    Parsing function to take care of type casting and shape changing
    """

    def _parse_fn(self, image, label):
        
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        
        label = tf.cast(label, tf.int32)
        label = tf.one_hot(label, 10)
        
        return image, label
    
    """
    Creating the training and testing dataset
    """
    def _create_dataset(self):
        
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        print('x_train.shape {}, y_train.shape {}, x_test.shape {}, y_test.shape{}'.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

        ### Train dataset
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_ds = train_ds.map(self._parse_fn)
        train_ds = train_ds.shuffle(buffer_size=60000).repeat(self._num_epoch)
        train_ds = train_ds.batch(self._batch_size)
        
        self._itr_per_epoch_train = x_train.shape[0] // self._batch_size
        
        ### testation dataset
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_ds = test_ds.map(self._parse_fn).batch(self._batch_size)
        
        self._itr_per_epoch_test = x_test.shape[0] // self._batch_size

        return train_ds, test_ds

    """
    Returns the logits of the network
    """
    def _model_fc(self, input):
        
        net = tf.layers.flatten(input)
        net = tf.layers.dense(net, units=128)
        net = tf.layers.dense(net, units=64)
        net = tf.layers.dense(net, units=32)
        net = tf.layers.dense(net, units=10)
        net = tf.nn.softmax(net)
        
        return net

    """
    Dataset iterator creation, training loop and also model summaries.
    """

    def fit(self):
        
        ### Dataset creation
        train_ds, test_ds = self._create_dataset()
        
        iter = tf.data.Iterator.from_structure(train_ds.output_types, train_ds.output_shapes)
        
        train_init_op = iter.make_initializer(train_ds)
        test_init_op = iter.make_initializer(test_ds)
        
        image, label = iter.get_next()        
        print('image_train.shape {}, label_train.shape {}'.format(image.get_shape(), label.get_shape()))
        
             
        ### Logits
        self._logits = self._model_fc(image)
        
        ### Loss and accuracy
        self._loss= tf.losses.softmax_cross_entropy(label, self._logits)
        
        true_pred  = tf.argmax(label, 1)
        model_pred = tf.argmax(self._logits, 1)
        
        self._acc = tf.cast(tf.equal(true_pred, model_pred), tf.float32)
        self._acc = tf.reduce_mean(self._acc)
        
        ### Optimizer
        self._train_op = tf.train.AdamOptimizer(self._lr).minimize(self._loss)
        
        ### Session
        sess = tf.Session()

        ### Initialization        
        init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]        
        sess.run(init_op)
        
        
        ### Summaries
        loss_summary = tf.summary.scalar('cross_entropy', self._loss)
        top1_acc_summary = tf.summary.scalar('top1_acc', self._acc)
        merged = tf.summary.merge([loss_summary, top1_acc_summary])
        
        train_writer = tf.summary.FileWriter('./train')
        test_writer  = tf.summary.FileWriter('./test')
        
        ### Training loop
        itr_train = 0 
        itr_test = 0
        test_every_epoch = 1
        
        for epoch in range(self._num_epoch):
            
            #Switch to train set
            sess.run(train_init_op)
            
            for itr in range(self._itr_per_epoch_train):
            
                _, _loss, acc, _image, _label, train_summary = sess.run([self._train_op, self._loss, self._acc, image, label, merged])
            
                train_writer.add_summary(train_summary, itr_train)
                itr_train = itr_train + 1

                if itr % 10 == 0:
                    print('Epoch: {0} Itr: {1} Loss:  {2:3.2f}, acc: {3:3.2f}'.format(epoch, itr, _loss, acc))
            
            ##testation every epoch
            if epoch % test_every_epoch == 0:
                
                #Switch to test set
                sess.run(test_init_op)
                test_loss = []
                test_acc = []
                
                
                while(True):
                    
                    try:
                        _loss, _acc, _image, _label, test_summary = sess.run([self._loss, self._acc, image, label, merged])
                        
                        test_writer.add_summary(test_summary, itr_train)
                        
                        test_loss.append(_loss)
                        test_acc.append(_acc)
                    
                    except tf.errors.OutOfRangeError:
                        
                        test_loss = np.asarray(test_loss)
                        test_loss = np.mean(test_loss)
                        
                        test_acc = np.asarray(test_acc)
                        test_acc = np.mean(test_acc)
                        
                        print('****Epoch: {0}, test loss:  {1:3.2f}, test acc:  {2:3.2f}'.format(epoch, test_loss, test_acc))
                                                
                        break
                    
    

def main(args):
    tf_lin = tf_FC()
    tf_lin.fit()
        

if __name__ == "__main__":
    tf.app.run(main=main)

    
    