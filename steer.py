#!/usr/bin/env python3

import os
import tensorflow as tf

LOGDIR = './save'

HIDDEN_LAYER_DEPTH = 1164
TRAIN_DROPOUT = 0.8
TEST_DROPOUT = 1.0
LEARNING_RATE = 1e-6 

class steer_nn():
    """
    Neural network to predict steering wheel turning angle. 
    """

#    @profile
    def __init__(self):
        self.create_nn()
        self.create_tensorflow()


        self.session = tf.InteractiveSession()

        # Merge all summaries and write them out
        self.merged_summaries = tf.merge_all_summaries()
        self.train_writer = tf.train.SummaryWriter("./tmp/train",self.session.graph)
        # Init session
        tf.initialize_all_variables().run()

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()

    def weight_variable(self,shape,stddev=0.1):
        initial = tf.truncated_normal(shape,stddev=stddev)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, writh bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1,strides,strides,1], padding='VALID')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    # Convolutional NN impl.
    def create_nn(self):
        with tf.name_scope("input_layer"):
            # input layer - [batch size, 45 h, 160 w, 3 channels]
            #self.img_in = tf.placeholder(tf.float32, [None, 66, 200, 3])
            self.img_in = tf.placeholder(tf.float32, [None, 45, 160, 3])
            tf.histogram_summary("input_img_in", self.img_in);

        # conv layers
        with tf.name_scope("conv_layer1"):
            Wc1 = self.weight_variable([5,5,3,24])
            tf.histogram_summary("conv1_Wc", Wc1);
            bc1 = self.bias_variable([24])
            tf.histogram_summary("conv1_bc", bc1);
            conv_layer1 = self.conv2d(self.img_in,Wc1,bc1,2)
            tf.histogram_summary('conv1_activation',conv_layer1)

        with tf.name_scope("conv_layer2"):
            Wc2 = self.weight_variable([5,5,24,36])
            tf.histogram_summary("conv2_Wc", Wc2);
            bc2 = self.bias_variable([36])
            tf.histogram_summary("conv2_bc", bc2);
            conv_layer2 = self.conv2d(conv_layer1,Wc2,bc2,2)
            tf.histogram_summary('conv2_activation',conv_layer2)

        with tf.name_scope("conv_layer3"):
            Wc3 = self.weight_variable([5,5,36,48])
            tf.histogram_summary("conv3_Wc", Wc3);
            bc3 = self.bias_variable([48])
            tf.histogram_summary("conv3_bc", bc3);
            conv_layer3 = self.conv2d(conv_layer2,Wc3,bc3,2)
            tf.histogram_summary('conv3_activation',conv_layer3)

        with tf.name_scope("conv_layer4"):
            Wc4 = self.weight_variable([3,3,48,64])
            tf.histogram_summary("conv4_Wc", Wc4);
            bc4 = self.bias_variable([64])
            tf.histogram_summary("conv4_bc", bc4);
            conv_layer4 = self.conv2d(conv_layer3,Wc4,bc4,1)
            tf.histogram_summary('conv4_activation',conv_layer4)

        #with tf.name_scope("conv_layer5"):
        #    Wc5 = self.weight_variable([3,3,64,64])
        #    bc5 = self.bias_variable([64])
        #    conv_layer5 = self.conv2d(conv_layer4,Wc5,bc5,1)

        with tf.name_scope("fc_layer1"):
            # Fully connected layer
            #Wfc1 = self.weight_variable([1152,HIDDEN_LAYER_DEPTH])
            Wfc1 = self.weight_variable([960,HIDDEN_LAYER_DEPTH])
            tf.histogram_summary("fc1_Wfc", Wfc1);
            bfc1 = self.bias_variable([HIDDEN_LAYER_DEPTH])
            tf.histogram_summary("fc1_bfc", bfc1);
            #conv_layer5_flat = tf.reshape(conv_layer5,[-1,1152])
            conv_layer5_flat = tf.reshape(conv_layer4,[-1,960])
            fc_layer1 = tf.nn.relu(tf.matmul(conv_layer5_flat,Wfc1) + bfc1)
            tf.histogram_summary('fc1__activation1',fc_layer1)
            self.keep_prob = tf.placeholder(tf.float32)
            fc_layer1_drop = tf.nn.dropout(fc_layer1,self.keep_prob)
            tf.histogram_summary('fc1_dropout_activation1',fc_layer1_drop)

        with tf.name_scope("fc_layer2"):
            # Fully connected layer
            Wfc2 = self.weight_variable([HIDDEN_LAYER_DEPTH,100])
            tf.histogram_summary("fc2_Wfc", Wfc2);
            bfc2 = self.bias_variable([100])
            tf.histogram_summary("fc2_bfc", bfc2);
            fc_layer2 = tf.nn.relu(tf.matmul(fc_layer1_drop,Wfc2) + bfc2)
            tf.histogram_summary('fc2_activation2',fc_layer2)
            fc_layer2_drop = tf.nn.dropout(fc_layer2,self.keep_prob)
            tf.histogram_summary('fc2_dropout_activation2',fc_layer2_drop)

        with tf.name_scope("fc_layer3"):
            # Fully connected layer
            Wfc3 = self.weight_variable([100,50])
            tf.histogram_summary("fc3_Wfc", Wfc3);
            bfc3 = self.bias_variable([50])
            tf.histogram_summary("fc3_bfc", bfc3);
            fc_layer3 = tf.nn.relu(tf.matmul(fc_layer2_drop,Wfc3) + bfc3)
            tf.histogram_summary('fc3_activation',fc_layer3)
            fc_layer3_drop = tf.nn.dropout(fc_layer3,self.keep_prob)
            tf.histogram_summary('fc3_dropout_activation',fc_layer3_drop)

        with tf.name_scope("fc_layer4"):
            # Fully connected layer
            Wfc4 = self.weight_variable([50,10])
            tf.histogram_summary("fc4_Wfc", Wfc4);
            bfc4 = self.bias_variable([10])
            tf.histogram_summary("fc4_bfc", bfc4);
            fc_layer4 = tf.nn.relu(tf.matmul(fc_layer3_drop,Wfc4) + bfc4)
            tf.histogram_summary('fc4_activation',fc_layer4)
            fc_layer4_drop = tf.nn.dropout(fc_layer4,self.keep_prob)
            tf.histogram_summary('fc4_dropout_activation',fc_layer4_drop)

        with tf.name_scope("output_layer"):
            # Output  
            Wout = self.weight_variable([10,1])
            tf.histogram_summary("out_W", Wout);
            bout = self.bias_variable([1])
            tf.histogram_summary("out_b", bout);
            self.predict_angle = tf.matmul(fc_layer4_drop,Wout) + bout
            tf.histogram_summary('out_pred_angle',self.predict_angle)

        # Image summaries
        tf.image_summary("Input image", self.img_in, max_images=20)

        layer1_image1 = tf.transpose(conv_layer1[0:1,:,:,:], perm=[3,1,2,0])
        layer1_combine_1 = tf.concat(2, layer1_image1)
        list_lc1 = tf.split(0,24,layer1_combine_1)
        layer1_combine_1= tf.concat(1, list_lc1)
        tf.image_summary("Convolution layer 1", layer1_combine_1, max_images=20)
        #tf.image_summary("Convolution layer 1", tf.reshape(conv_layer1[0,:,:,:], [24,21,78,1]), max_images=100)

        layer2_image1 = tf.transpose(conv_layer2[0:1,:,:,:], perm=[3,1,2,0])
        layer2_combine_1 = tf.concat(2, layer2_image1)
        list_lc2 = tf.split(0,36,layer2_combine_1)
        layer2_combine_1= tf.concat(1, list_lc2)
        tf.image_summary("Convolution layer 2", layer2_combine_1, max_images=20)

    def create_tensorflow(self):
        self.angle_truth = tf.placeholder(tf.float32, [None,1])
        self.cost = tf.reduce_mean(tf.square(self.angle_truth - self.predict_angle))
        # Monitor the cost of training
        tf.scalar_summary('Cost',self.cost)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.cost)

    def train(self,xs,ys,i):
        summary, _, loss = self.session.run(
                    [self.merged_summaries, self.optimizer, self.cost], 
                    feed_dict={
                        self.img_in: xs, 
                        self.angle_truth: ys, 
                        self.keep_prob: TEST_DROPOUT
                    })

        # Record summary every N batches
        if i % 10 == 0:
            self.train_writer.add_summary(summary,i)

        return loss

    def val(self,xs,ys,i):
        print("val: step %d, val loss %g"%(i, self.cost.eval(
                     feed_dict={
                        self.img_in: xs, 
                        self.angle_truth: ys, 
                        self.keep_prob: TRAIN_DROPOUT
                    })))

    def saveParam(self):
        if not os.path.exists(LOGDIR):
            os.makedirs(LOGDIR)
        checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
        filename = self.saver.save(self.session, checkpoint_path)
        print("Model saved in file: %s" % filename)

    def restoreParam(self):
        if not os.path.exists(LOGDIR):
            os.makedirs(LOGDIR)
        checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
        self.saver.restore(self.session, checkpoint_path)
        print("Model restored from file: %s" % checkpoint_path)


