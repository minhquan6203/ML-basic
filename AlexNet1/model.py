from audioop import bias
import tensorflow as tf

class AlexNet:
    def __init__(self,input_with=227, input_height=227,
                input_channels=3, num_class=1000, lr=0.01,momentum=0.9,
                keep_prop=0.5 ):
        self.input_with=input_with
        self.input_weight=input_height
        self.input_channels=input_channels
        self.num_class=num_class
        self.lr=lr
        self.momemtum=momentum
        self.keep_prob=keep_prop
        self.random_mean=0
        self.random_stddc=0.01
        #labels
        with tf.compat.v1.name_scope('labels'):
            self.Y=tf.compat.v1.placeholder(dtype=tf.compat.v1.float32, shape=[None, self.num_classes], name='Y')

        #dropout keep prob
        with tf.compat.v1.name_scope('dropout'):
            self.dropout_keep_prob=tf.compat.v1.placeholder(dtype=tf.compat.v1.float32,shape=(),name='dropout_keep_prob')
        
        #layer1
        with tf.compat.v1.name_scope('layer1'):
            layer1_activations=self.__conv(input=self.X,filter_with=11,filter_height=11,
                            filter_count=96,stride_x=4,stride_y=4,padding='VALID',init_bias_with_constant_1=False)
            layer1_lrn=self.__local_respone_normalization(input=layer1_activations)
            layer1_pool=self.__max_pool(input=layer1_lrn, filter_width=3, filter_height=3, stride_x=2, stride_y=2,padding='VALID')


        #layer2
        with tf.compat.v1.name_scope('layer2'):
            layer2_activations=self.__conv(input=self.X,filter_with=5,filter_height=5,
                            filter_count=256,stride_x=1,stride_y=1,padding='SAME',init_bias_with_constant_1=True)
            layer2_lrn=self.__local_respone_normalization(input=layer2_activations)
            layer2_pool=self.__max_pool(input=layer2_lrn, filter_width=3, filter_height=3, stride_x=2, stride_y=2,padding='VALID')

        #layer3
        with tf.compat.v1.name_scope('layer3'):
            layer3_activations = self.__conv(input=layer2_pool, filter_width=3, filter_height=3, 
                            filters_count=384,stride_x=1, stride_y=1, padding='SAME',init_biases_with_the_constant_1=False)

        #layer4
        with tf.compat.v1.name_scope('layer4'):
            layer4_activations = self.__conv(input=layer3_activations, filter_width=3, filter_height=3,
                            filters_count=384, stride_x=1, stride_y=1, padding='SAME',
                            init_biases_with_the_constant_1=True)

        #layer5
        with tf.compat.v1.name_scope('layer5'):
            layer5_activations = self.__conv(input=layer4_activations, filter_width=3, filter_height=3,
                                             filters_count=256, stride_x=1, stride_y=1, padding='SAME',
                                             init_biases_with_the_constant_1=True)
            layer5_pool = self.__max_pool(input=layer5_activations, filter_width=3, filter_height=3, stride_x=2,
                                          stride_y=2, padding='VALID')

        #layer6
        with tf.compat.v1.name_scope('layer6'):
            pool5_shape = layer5_pool.get_shape().as_list()
            flattened_input_size = pool5_shape[1] * pool5_shape[2] * pool5_shape[3]
            layer6_fc = self.__fully_connected(input=tf.reshape(layer5_pool, shape=[-1, flattened_input_size]),
                                               inputs_count=flattened_input_size, outputs_count=4096, relu=True,
                                               init_biases_with_the_constant_1=True)
            layer6_dropout = self.__dropout(input=layer6_fc)

        #layer7
        with tf.compat.v1.name_scope('layer7'):
            layer7_fc = self.__fully_connected(input=layer6_dropout, inputs_count=4096, outputs_count=4096, relu=True,
                                               init_biases_with_the_constant_1=True)
            layer7_dropout = self.__dropout(input=layer7_fc)

        #layer8
        with tf.compat.v1.name_scope("layer8"):
            layer8_logits = self.__fully_connected(input=layer7_dropout, inputs_count=4096,
                                                   outputs_count=self.num_classes, relu=False, name='logits')
        
        #cross entropy
        with tf.compat.v1.name_scope('cross_entropy'):
            cross_entropy=tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
    
            )
            pass

        #training
        with tf.compat.v1.name_scope('training'):
            loss_operation=tf.compat.v1.reduce_mean(cross_entropy,name='loss_operation')
            tf.compat.v1.summary.scalar(name='loss',tensor=loss_operation)
            optimizer=tf.compat.v1.train.MomentumOptimizer(learning_rate=self.lr,momentum=self.momemtum)
            grad_and_vars=optimizer.minimize(loss_operation,name='training_operation')
            self.training_operation=optimizer.apply_gradients(grad_and_vars,name='training_operation')
            for grad,var in grad_and_vars:
                if grad is not None:
                    with tf.compat.v1.name_scope(var.op.name + '/gradients'):
                        self.__variable_summaries(grad)
                    

        #accuracy
        with tf.compat.v2.name_scope("accuracy"):
            correct_prediction=tf.compat.v1.equal(tf.compat.v1.argmax(layer8_logits,1),
            tf.compat.v1.argmax(self.Y,1),name='correct_prediction')
            self.accuracy_operation=tf.compat.v1.reduce_mean(correct_prediction,tf.compat.v1.float32,
            name='accuracy_operation')
            tf.compat.v1.summary.scalar(name='accuracy',tensor=self.accuracy_operation)
        
    def train_epoch(self,sess,X_data,Y_data,batch_size=128,file_writer=None,summary_operation=None,epoch_number=None):
        num_examples=len(X_data)
        step=0
        sess=tf.compat.v1.Session()
        file_writer=tf.compat.v1.summary.FileWriter()
        for offset in range(0,num_examples,batch_size):
            end=offset+batch_size
            batch_x,batch_y=X_data[offset:end],Y_data[offset:end]
            if file_writer is not None and summary_operation is not None:
                _,summary=sess.run([self.training_operation,summary_operation],
                        feed_dict={self.X:batch_x,self.Y:batch_y,self.dropout_keep_prob:self.keep_prob})
                file_writer.add_summary(summary, epoch_number * (num_examples // batch_size + 1) + step)
                step += 1
            else:
                sess.run(self.training_operation, feed_dict={self.X: batch_x, self.Y: batch_y,
                                                             self.dropout_keep_prob: self.keep_prob})
    def evaluate(self,sess,X_data,Y_data,batch_size=128):
        num_examples=len(X_data)
        total_accuracy=0
        sess=tf.compat.v1.Session()
        for offset in range(0,num_examples,batch_size):
            end=offset+batch_size
            batch_x,batch_y=X_data[offset:end],Y_data[offset:end]
            batch_accuracy=sess.run(self.accuracy_operation,feed_dict={self.X: batch_x, self.Y: batch_y,
                                                                          self.dropout_keep_prob: 1.0})
            total_accuracy+=(batch_accuracy*len(batch_x))
        return total_accuracy/num_examples

    
    def save(self,sess,file_name):
        saver=tf.compat.v1.train.Saver()
        saver.save(sess, file_name)
    
    def restore(self,sess,checkpoint_dir):
        saver=tf.compat.v1.train.Saver()
        saver.restore(sess,tf.compat.v1.train.latest_checkpoint(checkpoint_dir))

    def __random_values(self, shape):
        return tf.random_normal(shape=shape, mean=self.random_mean, stddev=self.random_stddev, dtype=tf.float32)

    def __varible_summaries(self,var):
        mean=tf.compat.v1.reduce_mean(var)
        stddev=tf.compat.v1.sqrt(tf.compat.v1.reduce_mean(tf.compat.v1.square(var-mean)))
        tf.compat.v1.summary.scalar('min',tf.compat.v1.reduce_min(var))
        tf.compat.v1.summary.scalar('max',tf.compat.v1.reduce_max(var))
        tf.compat.v1.summary.scalar('mean',tf.compat.v1.reduce_mean(var))
        tf.compat.v1.summary.scalar('stddev',stddev)
        tf.compat.v1.summary.histogram('histogram',var)
    
    def __conv(self,input,filter_with,filter_height,filter_count,
            stride_x,stride_y,padding='VALID',init_bias_with_constant_1=False,name='conv'):
        with tf.compat.v1.name_scope(name):
            input_chanels=input.get_shape()[-1].value
            filters=tf.compat.v1.Variable(self.__random_values(shape=[filter_with,filter_height,input_chanels,filter_count]),name='filters')
            convs=tf.compat.v1.nn.conv2d(input=input,filter=filters,strides=[1,stride_x,stride_y,1],padding=padding,name='convs')
            if init_bias_with_constant_1:
                biases=tf.compat.v1.Variable(tf.compat.v1.ones(shape=[filter_count],dtype=tf.compat.v1.float32),name='biases')
            else:
                biases=tf.compat.v1.Variable(tf.compat.v1.zeros(shape=[filter_count],dtype=tf.compat.v1.float32),name='biases')
            preactivations = tf.nn.bias_add(convs, biases, name='preactivations')
            activations = tf.nn.relu(preactivations, name='activations')

            with tf.compat.v1.name_scope('filter_summaries'):
                self.__varible_summaries(filters)
            
            with tf.compat.v1.name_scope('bias_summaries'):
                self.__varible_summaries(biases)
            
            with tf.compat.v1.name_scope("preactivations_histogram"):
                tf.compat.v1.summary.histogram('preactivation',preactivations)
            
            with tf.compat.v1.name_scope('activation'):
                tf.compat.v1.summary.histogram('activation',activations)
            return activations
        
    def __local_respone_normalization(self,input,name='lrn'):
        with tf.compat.v1.name_scope(name):
            lrn=tf.compat.v1.nn.local_response_normalization(input=input,depth_radius=2,alpha=1e-4,beta=0.75, name='local_response_normalization')
            return lrn
    def __max_pool(self,input,filter_with,filter_height,stride_x,stride_y,padding='VALID', name='pool'):
        with tf.compat.v1.name_scope(name):
            pool=tf.compat.v1.nn.max_pool(input=input,ksize=[1,filter_height,filter_with,1],strides=[1, stride_y, stride_x, 1],padding=padding,name='pool')
    def __fully_connected(self,input,input_count,output_count,relu=True,init_bias_with_constant_1=False,name='fully_connected'):
        with tf.compat.v1.name_scope(name):
            wights=tf.compat.v1.Variable(self.__random_values(shape=[input_count,output_count]),name='wights')
            if init_bias_with_constant_1:
                biases=tf.compat.v1.Variable(tf.compat.v1.ones(shape=[output_count],dtype=tf.compat.v1.float32),name='biases')
            else:
                biases=tf.compat.v1.Variable(tf.compat.v1.zeros(shape=[output_count],dtype=tf.compat.v1.float32),name='biases')
            preactivations = tf.nn.bias_add(tf.compat.v1.matmul(input,wights), biases, name='preactivations')

            if relu:
                activations = tf.nn.relu(preactivations, name='activations')

            with tf.compat.v1.name_scope('wight_summaries'):
                self.__varible_summaries(wights)
            
            with tf.compat.v1.name_scope('bias_summaries'):
                self.__varible_summaries(biases)
    
            with tf.compat.v1.name_scope("preactivations_histogram"):
                tf.compat.v1.summary.histogram('preactivation',preactivations)
            
            if relu:
                with tf.compat.v1.name_scope("activations_histogram"):
                    tf.compat.v1.summary.histogram('activation',activations)
            
            if relu:
                return activations
            else:
                return activations
            
    def __dropout(self,input,name='dropout'):
        with tf.compat.v1.name_scope(name):
            return tf.compat.v1.nn.dropout(input,keep_prob=self.dropout_keep_prob,name='dropout')
