import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

class tf_FC(object):

    def __init__(self):
        
        self._lr = 0.01
        self._batch_size = 60000
        self._num_epoch = 100
        
        self._loss_train = np.zeros([(10000 // self._batch_size) * self._num_epoch , 1])


    def _parse_fn(self, image, label):
        
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        
        label = tf.cast(label, tf.int32)
        label = tf.one_hot(label, 10)
        
        return image, label

    def _create_dataset(self):
        
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        print('x_train.shape {}, y_train.shape {}, x_test.shape {}, y_test.shape{}'.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

        ### Train dataset
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_ds = train_ds.map(self._parse_fn)
        train_ds = train_ds.shuffle(buffer_size=60000)
        train_ds = train_ds.batch(self._batch_size).repeat(self._num_epoch)
        
        self._itr_per_epoch_train = x_train.shape[0] // self._batch_size
        
        ### Validation dataset
        val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        val_ds = val_ds.map(self._parse_fn).batch(self._batch_size)
        
        self._itr_per_epoch_valid = x_test.shape[0] // self._batch_size

        return train_ds, val_ds

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

    def fit(self):
        
        ### Dataset creation
        train_ds, val_ds = self._create_dataset()
        
        train_itr = train_ds.make_one_shot_iterator()
        image_train, label_train = train_itr.get_next()        
        print('image_train.shape {}, label_train.shape {}'.format(image_train.get_shape(), label_train.get_shape()))
        
        val_itr = val_ds.make_one_shot_iterator()
        image_val, label_val = val_itr.get_next()
        print('image_val.shape {}, label_val.shape {}'.format(image_val.get_shape(), label_val.get_shape()))
        
        ### Logits
        self._logits = self._model_fc(image_train)
        
        ### Loss and accuracy
        self._loss= tf.losses.softmax_cross_entropy(label_train, self._logits)
        
        true_pred  = tf.argmax(label_train, 1)
        model_pred = tf.argmax(self._logits, 1)
        
        self._acc = tf.cast(tf.equal(true_pred, model_pred), tf.float32)
        self._acc = tf.reduce_mean(self._acc)
        
        ## Validation loss and accuracy
        self._val_logits = self._model_fc(image_val)
        self._val_loss   = tf.losses.softmax_cross_entropy(label_val, self._val_logits)
        self._val_acc    = tf.reduce_mean(tf.cast( tf.equal(tf.argmax(label_val, 1), tf.argmax(self._val_logits, 1)) , tf.float32))
        
        ### Optimizer
        self._train_op = tf.train.AdamOptimizer(self._lr).minimize(self._loss)
        
        ### Session
        sess = tf.Session()

        ### Initialization        
        init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]        
        sess.run(init_op)
        
        
        ### Summaries
        tf.summary.scalar('cross_entropy', self._loss)
        tf.summary.scalar('training_top1', self._acc)
        tf.summary.scalar('val_loss', self._val_loss)
        tf.summary.scalar('val_acc', self._val_acc)
        merged_summaries = tf.summary.merge_all()
        
        train_writer = tf.summary.FileWriter('./train')
        val_writer   = tf.summary.FileWriter('./val')
        
        ### Training loop
        loss = []
        count_train = 0 
        count_val = 0
        valid_every_epoch = 1
        
        for epoch in range(self._num_epoch):
            for itr in range(self._itr_per_epoch_train):
            
                merged, _, _loss, acc, image, label = sess.run([merged_summaries, self._train_op, self._loss, self._acc, image_train, label_train])
            
                train_writer.add_summary(merged, count_train)
                count_train = count_train + 1
            
                ##self._loss_train[itr] = _loss
                loss.append(_loss)
                if itr % 10 == 0:
                    print('Epoch: {0} Itr: {1} Loss:  {2:3.2f}, acc: {3:3.2f}'.format(epoch, itr, _loss, acc))
            
            ##Validation every epoch
            if epoch % valid_every_epoch == 0:
                _loss, _acc = sess.run([self._val_loss, self._val_acc])
                print('Epoch: {0}, Val loss:  {1:3.2f}, Val acc:  {2:3.2f}'.format(epoch, _loss, _acc))
                val_writer.add_summary(merged, count_val)
                count_val += 1
        
        loss = np.asanyarray(loss)
        plt.plot(np.arange(loss.shape[0]), loss)
        plt.show()
        

if __name__ == "__main__":
    
    tf_lin = tf_FC()
    tf_lin.fit()
    
    