'''
Tensorflow implementation of Attentional Factorization Machines (AFM)

@author: 
Xiangnan He (xiangnanhe@gmail.com)
Hao Ye (tonyfd26@gmail.com)

@references:
'''
import math
import os, sys
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, log_loss
from time import time
import argparse
import LoadData as DATA
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

#################### Arguments ####################

def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepAFM.")
    parser.add_argument('--process', nargs='?', default='train',
                        help='Process type: train, evaluate.')
    parser.add_argument('--path', nargs='?', default='../data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-tag',
                        help='Choose a dataset.')
    parser.add_argument('--valid_dimen', type=int, default=10,
                        help='Valid dimension of the dataset. (e.g. frappe=10, ml-tag=3)')
    parser.add_argument('--epoch', type=int, default=20,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=-1,
                        help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save to pretrain file; 2: initialize from pretrain and save to pretrain file')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size.')

    parser.add_argument('--field_size',type=int, default = 21,help='Number of fields.')
    parser.add_argument('--dropout_deep', nargs='?', default='[0.5, 0.5, 0.5]',
                        help='Dropout in DNN.') 
    parser.add_argument('--deep_layers', nargs='?', default='[32, 32]',
                        help='Deep layers in DNN.') 

    parser.add_argument('--attention', type=int, default=1,
                        help='flag for attention. 1: use attention; 0: no attention')
    parser.add_argument('--hidden_factor', nargs='?', default='[16,16]',
                        help='Number of hidden factors.')
    parser.add_argument('--lamda_attention', type=float, default=1e+2,
                        help='Regularizer for attention part.')
    parser.add_argument('--keep', nargs='?', default='[1.0,0.5]',
                        help='Keep probility (1-dropout) of each layer. 1: no dropout. The first index is for the attention-aware pairwise interaction layer.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate.')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to show the performance of each epoch (0 or 1)')
    parser.add_argument('--batch_norm', type=int, default=0,
                    help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--decay', type=float, default=0.999,
                    help='Decay value for batch norm')
    parser.add_argument('--activation', nargs='?', default='relu',
                    help='Which activation function to use for deep layers: relu, sigmoid, tanh, identity')
    parser.add_argument('--gpu', type=str, default='0',
                    help='gpu to run this program (0 or 1)')

    return parser.parse_args()

class DeepAFM(BaseEstimator, TransformerMixin):
    def __init__(self, features_M, pretrain_flag, save_file, 
        hidden_factor, valid_dimension, activation_function,
        epoch, batch_size, learning_rate, lamda_attention, keep,
        optimizer_type, batch_norm, decay, verbose, field_size, random_seed=2016,
        i_gpu='0',dropout_deep=[0.5, 0.5, 0.5], deep_layers=[32, 32]):
        # bind params to class
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_factor = hidden_factor
        self.valid_dimension = valid_dimension
        self.activation_function = activation_function
        self.save_file = save_file
        self.pretrain_flag = pretrain_flag
        self.features_M = features_M
        self.lamda_attention = lamda_attention
        self.keep = keep
        self.epoch = epoch
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.decay = decay
        self.verbose = verbose
        ##
        self.i_gpu = i_gpu
        self.dropout_deep = dropout_deep
        self.deep_layers = deep_layers
        self.field_size = field_size
        self.embedding_size = hidden_factor[1]
        self.deep_layers_activation = tf.nn.relu
        ##
        # performance of each epoch
        self.train_rmse, self.valid_rmse, self.test_rmse = [], [], []

        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.train_features = tf.placeholder(
                tf.int32, shape=[None, None], name="train_features_afm")  # None * features_M
            self.train_labels = tf.placeholder(
                tf.float32, shape=[None, 1], name="train_labels_afm")  # None * 1
            self.dropout_keep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_afm")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase_afm")
            self.dropout_keep_deep = tf.placeholder(
                tf.float32, shape=[None], name="dropout_keep_deep")
            # Variables.
            self.weights = self._initialize_weights()

            # Model.
            self.nonzero_embeddings = tf.nn.embedding_lookup(
                    self.weights['feature_embeddings'], self.train_features) # None * M' * K
            
            element_wise_product_list = []
            count = 0
            for i in range(0, self.valid_dimension):
                for j in range(i+1, self.valid_dimension):
                    element_wise_product_list.append(tf.multiply(self.nonzero_embeddings[:,i,:], self.nonzero_embeddings[:,j,:]))
                    count += 1
            self.element_wise_product = tf.stack(element_wise_product_list) # (M'*(M'-1)) * None * K
            self.element_wise_product = tf.transpose(self.element_wise_product, perm=[1,0,2], name="element_wise_product") # None * (M'*(M'-1)) * K
            self.interactions = tf.reduce_sum(self.element_wise_product, 2, name="interactions")
            # _________ MLP Layer / attention part _____________
            num_interactions = self.valid_dimension*(self.valid_dimension-1)/2
            self.attention_mul = tf.reshape(tf.matmul(tf.reshape(self.element_wise_product, shape=[-1, self.hidden_factor[1]]), \
                   self.weights['attention_W']), shape=[-1, num_interactions, self.hidden_factor[0]])
            self.attention_exp = tf.exp(tf.reduce_sum(tf.multiply(self.weights['attention_p'], tf.nn.relu(self.attention_mul + \
                    self.weights['attention_b'])), 2, keep_dims=True)) # None * (M'*(M'-1)) * 1
            self.attention_sum = tf.reduce_sum(self.attention_exp, 1, keep_dims=True) # None * 1 * 1
            self.attention_out = tf.div(self.attention_exp, self.attention_sum, name="attention_out") # None * (M'*(M'-1)) * 1
            self.attention_out = tf.nn.dropout(self.attention_out, self.dropout_keep[0]) # dropout
            
            # _________ Attention-aware Pairwise Interaction Layer _____________
            self.AFM = tf.reduce_sum(tf.multiply(self.attention_out,\
                    self.element_wise_product), 1, name="afm") # None * K
            self.AFM_FM = tf.reduce_sum(self.element_wise_product, 1, name="afm_fm") # None * K
            self.AFM_FM = self.AFM_FM / num_interactions
            self.AFM = tf.nn.dropout(self.AFM, self.dropout_keep[1]) # dropout

            # ---------- Deep component ----------
            self.y_deep = tf.reshape(self.nonzero_embeddings, shape=[-1, self.field_size * self.embedding_size]) # None * (F*K)
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])
            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" %i]), self.weights["bias_%d"%i]) # None * layer[i] * 1
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn="bn_%d" %i) # None * layer[i] * 1
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[1+i]) # dropout at each Deep layer

            # ---------- DeepAFM ----------
            self.Feature_bias = tf.reduce_sum(tf.nn.embedding_lookup(\
                                self.weights['feature_bias'],self.train_features),2)  # None * F
            concat_input = tf.concat([self.Feature_bias, self.AFM, self.y_deep], axis=1)
            self.out = tf.add(tf.matmul(
                concat_input, self.weights["concat_projection"]), self.weights["concat_bias"])

            # Compute the loss.
            self.out = tf.nn.sigmoid(self.out, name='DeepAFM_out')
            if self.lamda_attention > 0:
                self.loss = tf.losses.log_loss(self.train_labels, self.out) + \
                tf.contrib.layers.l2_regularizer(self.lamda_attention)(self.weights['attention_W'])  # regulizer
            else:
                self.loss = tf.losses.log_loss(self.train_labels, self.out)

            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,\
                                     beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,\
                                             initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(\
                    learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(\
                    learning_rate=self.learning_rate, momentum=0.95).minimize(self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape() # shape is an array of tf.Dimension
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print "#params: %d" %total_parameters 
    
    def _init_session(self):
        # adaptively growing video memory
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list=self.i_gpu
        return tf.Session(config=config)

    def _initialize_weights(self):
        all_weights = dict()
        if self.pretrain_flag > 0:
            from_file = self.save_file
            from_file = self.save_file.replace('deep_afm', 'fm')
            weight_saver = tf.train.import_meta_graph(from_file + '.meta')
            pretrain_graph = tf.get_default_graph()
            feature_embeddings = pretrain_graph.get_tensor_by_name('feature_embeddings:0')
            feature_bias = pretrain_graph.get_tensor_by_name('feature_bias:0')
            with self._init_session() as sess:
                weight_saver.restore(sess, from_file)
                fe, fb = sess.run([feature_embeddings, feature_bias])
            
            all_weights['feature_embeddings'] = tf.Variable(fe, dtype=tf.float32, name='feature_embeddings')
            all_weights['feature_bias'] = tf.Variable(fb, dtype=tf.float32, name='feature_bias')
        else:
            all_weights['feature_embeddings'] = tf.Variable(
                tf.random_normal([self.features_M, self.hidden_factor[1]], 0.0, 0.01),
                name='feature_embeddings')  # features_M * K
            all_weights['feature_bias'] = tf.Variable(
                tf.random_uniform([self.features_M, 1], 0.0, 0.0), name='feature_bias')  # features_M * 1

        # deep layers
        num_layer = len(self.deep_layers)
        input_size = self.field_size * self.embedding_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        all_weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        all_weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                                        dtype=np.float32)  # 1 * layers[0]
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))
            all_weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i-1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            all_weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]

        # final concat projection layer
        input_size = self.field_size + self.embedding_size + self.deep_layers[-1]
        glorot = np.sqrt(2.0 / (input_size + 1))
        all_weights["concat_projection"] = tf.Variable(
                        np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                        dtype=np.float32)  # layers[i-1]*layers[i]
        all_weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        # attention
        glorot = np.sqrt(2.0 / (self.hidden_factor[0]+self.hidden_factor[1]))
        all_weights['attention_W'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.hidden_factor[1], self.hidden_factor[0])), dtype=np.float32, name="attention_W")  # K * AK
        all_weights['attention_b'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.hidden_factor[0])), dtype=np.float32, name="attention_b")  # 1 * AK
        all_weights['attention_p'] = tf.Variable(
                np.random.normal(loc=0, scale=1, size=(self.hidden_factor[0])), dtype=np.float32, name="attention_p") # AK

        return all_weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.decay, center=True, scale=True, updates_collections=None,
            is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.decay, center=True, scale=True, updates_collections=None,
            is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, data):  # fit a batch
        feed_dict = {
                        self.train_features: data['X'],
                        self.train_labels: data['Y'], 
                        self.dropout_keep: self.keep, 
                        self.train_phase: True,
                        self.dropout_keep_deep: self.dropout_deep,
                        }
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def get_random_block_from_data(self, data, batch_size):  # generate a random block of training data
        start_index = np.random.randint(0, len(data['Y']) - batch_size)
        X , Y = [], []
        # forward get sample
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i + 1
            else:
                break
        # backward get sample
        i = start_index
        while len(X) < batch_size and i >= 0:
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i - 1
            else:
                break
        return {'X': X, 'Y': Y}
    
    def get_ordered_block_from_data(self, data, batch_size, index):  # generate a ordered block of data
        start_index = index*batch_size
        X , Y = [], []
        # get sample
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append(data['Y'][i])
                X.append(data['X'][i])
                i = i + 1
            else:
                break
        return {'X': X, 'Y': Y}

    def shuffle_in_unison_scary(self, a, b): # shuffle two lists simutaneously
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    def train(self, Train_data, Validation_data, Test_data):  # fit a dataset
        # Check Init performance
        if self.verbose > 0:
            t2 = time()
            init_train = self.evaluate(Train_data)
            init_valid = self.evaluate(Validation_data)
            print("Init: \t train=%.4f, validation=%.4f [%.1f s]" %(init_train, init_valid, time()-t2))

        
        
        for epoch in xrange(self.epoch):
            # fd = open('Deepafm_flush.out','a')
            t1 = time()
            self.shuffle_in_unison_scary(Train_data['X'], Train_data['Y'])
            total_batch = int(len(Train_data['Y']) / self.batch_size)
            for i in xrange(total_batch):
                # generate a batch
                batch_xs = self.get_random_block_from_data(Train_data, self.batch_size)
                # Fit training
                self.partial_fit(batch_xs)
            t2 = time()

            # evaluate training and validation datasets
            train_result = self.evaluate(Train_data)
            valid_result = self.evaluate(Validation_data)
            self.train_rmse.append(train_result)
            self.valid_rmse.append(valid_result)
            if self.verbose > 0 and epoch%self.verbose == 0:
                print("Epoch %d [%.1f s]\ttrain=%.4f, validation=%.4f [%.1f s]"
                      %(epoch+1, t2-t1, train_result, valid_result, time()-t2))
                # fd.write("%d:[%d] train-result=%.4f, valid-result=%.4f [%.1f s]\n"%(os.getpid(),epoch + 1, train_result, valid_result, time() - t1))
                # fd.flush()
                # fd.close()

            # test_result = self.evaluate(Test_data)
            # print("Epoch %d [%.1f s]\ttest=%.4f [%.1f s]"
            #       %(epoch+1, t2-t1, test_result, time()-t2))
            if self.eva_termination(self.valid_rmse):
                break

        if self.pretrain_flag < 0 or self.pretrain_flag == 2:
            print "Save model to file as pretrain."
            self.saver.save(self.sess, self.save_file)

    def eva_termination(self, vld):
        f = lambda x : -1 * x
        valid = map(f, vld)
        if len(valid) > 5:
            if valid[-1] > valid[-2] and valid[-2] > valid[-3] and valid[-3] > valid[-4] and valid[-4] > valid[-5]:
                return True
        return False

    def evaluate(self, data):  # evaluate the results for an input set
        num_example = len(data['Y'])
        # fetch the first batch
        batch_index = 0
        batch_xs = self.get_ordered_block_from_data(data, self.batch_size, batch_index)
        # batch_xs = data
        y_pred = None
        # if len(batch_xs['X']) > 0:
        while len(batch_xs['X']) > 0:
            num_batch = len(batch_xs['Y'])
            feed_dict = {
                            self.train_features: batch_xs['X'],
                            self.train_labels: [[y] for y in batch_xs['Y']],
                            self.dropout_keep: list(1.0 for i in range(len(self.keep))),
                            self.train_phase: False,
                            self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                            }
            a_exp, a_sum, a_out, batch_out = self.sess.run((self.attention_exp, self.attention_sum, self.attention_out, self.out), feed_dict=feed_dict)
            
            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))
            # fetch the next batch
            batch_index += 1
            batch_xs = self.get_ordered_block_from_data(data, self.batch_size, batch_index)

        y_true = np.reshape(data['Y'], (num_example,))
        ###
        # np.savetxt('y_pre',y_pred)
        # ###
        # predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
        # predictions_bounded = np.minimum(predictions_bounded, np.ones(num_example) * max(y_true))  # bound the higher values
        # RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
        # return RMSE
        return roc_auc_score(y_true, y_pred)

def make_save_file(args):
    pretrain_path = '../pretrain/deep_afm_%s_%d' %(args.dataset, eval(args.hidden_factor)[1])
    # if args.mla:
    #     pretrain_path += '_mla'
    if not os.path.exists(pretrain_path):
        os.makedirs(pretrain_path)
    save_file = pretrain_path+'/%s_%d' %(args.dataset, eval(args.hidden_factor)[1])
    return save_file

def train(args):
    # Data loading

    data = DATA.LoadData(args.path, args.dataset)
    if args.verbose > 0:
        print("DeepAFM: dataset=%s, factors=%s, valid_dim=%d, #epoch=%d, batch=%d, lr=%.4f, \
            lambda_attention=%.1e, keep=%s, optimizer=%s, batch_norm=%d, decay=%f, \
            activation=%s, field_size=%d, dropout_deep=%s, deep_layers=%s"
              %(args.dataset, args.hidden_factor, args.valid_dimen,args.epoch, args.batch_size, \
                args.lr, args.lamda_attention, args.keep, args.optimizer,\
              args.batch_norm, args.decay, args.activation, args.field_size,\
              args.dropout_deep, args.deep_layers))
    activation_function = tf.nn.relu
    if args.activation == 'sigmoid':
        activation_function = tf.sigmoid
    elif args.activation == 'tanh':
        activation_function == tf.tanh
    elif args.activation == 'identity':
        activation_function = tf.identity
    
    save_file = make_save_file(args)
    # Training
    t1 = time()

    model = DeepAFM(data.features_M, args.pretrain, save_file, eval(args.hidden_factor), args.valid_dimen, 
        activation_function,args.epoch, args.batch_size, args.lr, args.lamda_attention, eval(args.keep),
        args.optimizer, args.batch_norm, args.decay, args.verbose, args.field_size,random_seed=2016,
        i_gpu = args.gpu,dropout_deep=eval(args.dropout_deep),deep_layers=eval(args.deep_layers))
    
    model.train(data.Train_data, data.Validation_data, data.Test_data)
    
    # Find the best validation result across iterations
    best_valid_score = 0
    best_valid_score = max(model.valid_rmse)
    best_epoch = model.valid_rmse.index(best_valid_score)
    print ("Best Iter(validation)= %d\t train = %.4f, valid = %.4f [%.1f s]" 
           %(best_epoch+1, model.train_rmse[best_epoch], model.valid_rmse[best_epoch], time()-t1))

def evaluate(args):
    # load test data
    data = DATA.LoadData(args.path, args.dataset).Test_data
    save_file = make_save_file(args)
    
    # load the graph
    weight_saver = tf.train.import_meta_graph(save_file + '.meta')
    pretrain_graph = tf.get_default_graph()
    # load tensors 
    # feature_embeddings = pretrain_graph.get_tensor_by_name('feature_embeddings:0')
    # feature_bias = pretrain_graph.get_tensor_by_name('feature_bias:0')
    # bias = pretrain_graph.get_tensor_by_name('bias:0')
    # afm = pretrain_graph.get_tensor_by_name('afm:0')
    out_of_afm = pretrain_graph.get_tensor_by_name('DeepAFM_out:0')
    # placeholders for afm
    train_features_afm = pretrain_graph.get_tensor_by_name('train_features_afm:0')
    train_labels_afm = pretrain_graph.get_tensor_by_name('train_labels_afm:0')
    dropout_keep_afm = pretrain_graph.get_tensor_by_name('dropout_keep_afm:0')
    train_phase_afm = pretrain_graph.get_tensor_by_name('train_phase_afm:0')
    dropout_keep_deep = pretrain_graph.get_tensor_by_name('dropout_keep_deep:0')

    # restore session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    weight_saver.restore(sess, save_file)

    # start evaluation
    num_example = len(data['Y'])
    feed_dict = {
                    train_features_afm: data['X'],
                    train_labels_afm: [[y] for y in data['Y']],
                    dropout_keep_afm: [1.0,1.0],
                    dropout_keep_deep: [1.0]*3,
                    train_phase_afm: False
                }
    predictions = sess.run((out_of_afm), feed_dict=feed_dict)

    # calculate rmse
    y_pred_afm = np.reshape(predictions, (num_example,))
    y_true = np.reshape(data['Y'], (num_example,))
    
    auc_score = roc_auc_score(y_true, y_pred_afm)
    print("Test AUC: {:.6f}".format(auc_score))
    

if __name__ == '__main__':
    args = parse_args()

    # if args.mla:
    #     args.lr = 0.1
    #     args.keep = '[1.0,1.0]'
    #     args.lamda_attention = 10.0
    # else:
    #     args.lr = 0.1
    #     args.keep = '[1.0,0.5]'
    #     args.lamda_attention = 100.0

    if args.process == 'train':
        train(args)
    elif args.process == 'evaluate':
        evaluate(args)

