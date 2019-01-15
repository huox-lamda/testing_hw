import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import warnings
import theano.tensor.shared_randomstreams
from theano.tensor.nnet import conv
import sys
import time
import os
import os.path
import cPickle
import csv

from conv_net import * 
from process_transfer import *
#from data_split import *
from eval_results import *


warnings.filterwarnings("ignore")   

#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)

def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
    return updates

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)
    
def get_batch(train_code, train_report, train_label, index_i, batch_size):
     code_x = train_code[index_i*batch_size]
     report_x = train_report[index_i*batch_size:(index_i+1)*batch_size]
     for i in xrange(batch_size-1):
        code_x = np.vstack((code_x,train_code[index_i*batch_size+i+1]))
     y = train_label[index_i*batch_size:(index_i+1)*batch_size]
     
     return code_x, report_x, y


def get_testsets(test_code, test_report, test_label):
    code_x = test_code[0]
    for i in xrange(test_c-1):
        code_x = np.vstack((code_x,test_code[i+1]))
    report_x = test_report
    y = test_label
    return code_x, report_x, y

def dataCopy(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

def print_results( test_p, test_y, labels, test_index, project_name, target_name):
    tobecheckdir =  project_name+"_"+target_name+ "/transfer/"+str(repeat_index)+ "/results"
    if os.path.isdir(tobecheckdir)==False:
        os.makedirs(tobecheckdir)
    
    write_f = file(tobecheckdir+ "/results"+str(test_index)+".out",'w+')
    for score in test_p:
        write_f.write(str(score)+ "\n")
    write_f.close()
    
    write_f = file(tobecheckdir+ "/predictions"+str(test_index)+".out",'w+')
    for score in test_y:
        write_f.write(str(score)+ "\n")
    write_f.close()
    
    write_f = file(tobecheckdir + "/labels"+str(test_index)+".out",'w+')
    for score in labels:
        write_f.write(str(score)[1:-1]+ "\n")
    write_f.close()



    
def print_layer2_inputs( layer2_inputs):
     write_f = file("layer2_inputs.out",'w+')
     for node in layer2_inputs:
        write_f.write( str(node) + "\n" )

def get_parameters(argument):
    switcher = {
        0: [0],
        1: [0.1],
        2: [0.15],
        3: [0.2],
        4: [0.25],
        5: [0.30],
        6: [0.35],
        7: [0.4],
        8: [0.45],
        9: [0.5]        
    }
    return switcher.get(argument, "nothing")
    
if __name__=="__main__":
       all_time = time.time()  

       top1_task,top3_task,top5_task,top10_task,top20_task,acc_task,auc_task,map_task,mrr_task  = [], [], [], [], [], [], [], [], []
#       project_list = ["httpclient","jackrabbit","lucene-solr"]
#       target_list = ["httpclient","jackrabbit","lucene-solr"]
       project_list = ["aspectj"]
       target_list = ["jackrabbit"]
       for project_name in project_list:
         for target_name in target_list:
           if project_name==target_name: continue
           project_time = time.time()
           print project_name, target_name
           w2v_file = "word2vec.bin"
           top1_all = []
           top3_all = []
           top5_all = []
           top10_all = []
           top20_all = []
           acc_all = []
           auc_all = []
           map_all = []
           mrr_all = []
           index_num = 2
           for repeat_indexx in xrange(index_num):
              repeat_index = repeat_indexx+8
              print "index_num: ", str(repeat_index)
              code_maxl = 50 #code_maxl max statements per file
              code_maxk = 20 #code_maxk max words per statement

              if target_name == "httpclient": test_num = 51
              elif target_name == "jackrabbit": test_num = 427
              elif target_name == "lucene-solr": test_num = 156
              else: test_num = 310
#              test_num = 5 #test sets number
              
              report_maxl = 200;
              test_c = 300
              lr_decay = 0.95
              conv_non_linear = "conv"
              t_hidden_units = [100,2]
              s_hidden_units = [100,2]
              hidden_units = [100,2]
              t_filter_hs = [2,3,4]
              s_filter_hs = [2,3,4]
              n_epochs = 10
              sqr_norm_lim = 9
              batch_size = 20
              dropout_rate = [0.25]
#              dropout_rate = get_parameters(p1)
              
              if repeat_index==0:
                 write_time = file(project_name + "_" +target_name + "/transfer/settings.txt","w+")
    
                 parameters = [("token filter shape",t_filter_hs),("statement filter shape",s_filter_hs), ("hidden_units",hidden_units),
                      ("dropout", dropout_rate), ("batch_size",batch_size),("n_epochs", n_epochs),("index_num",index_num),("test_num",test_num),
                        ("learn_decay",lr_decay), ("conv_non_linear", conv_non_linear), ("sqr_norm_lim",sqr_norm_lim), ("test_c",test_c)]
                 print parameters 
                 write_time.write(str(parameters))
                 write_time.close()
         
             
             
              train_report, train_code, train_labels, target_report, target_code, target_labels, W = load_data(project_name, target_name, code_maxl, code_maxk, report_maxl, test_num, w2v_file, repeat_index)    
              cPickle.dump([train_report, train_code, train_labels, target_report, target_code, target_labels, W], open(project_name +"_"+target_name+"/transfer/"+str(repeat_index)+"/parameters.in", "wb"))
              print "Finish processing!"
        
        # begin training
              print "loading data for training"
              X = cPickle.load(open(project_name +"_"+target_name+"/transfer/"+str(repeat_index)+"/parameters.in","rb"))
              train_report, train_code, train_labels, target_report, target_code, target_labels, W = X[0], X[1], X[2], X[3], X[4], X[5], X[6]
              train_report = np.array(train_report,dtype="int")
              target_report = np.array(target_report, dtype="int")
              print "finish reading"
         
              labels = train_labels
              U = W           
              
              rng = np.random.RandomState(1234)   
              img_h = len(train_code[0][0])    
              img_w = 300
              activations=[Iden]
         
              stat_length = code_maxl
              t_filter_w = img_w
              t_feature_maps = t_hidden_units[0]
         
         
              t_filter_shapes = []
              t_pool_sizes = []
              for filter_h in t_filter_hs:
                  t_filter_shapes.append((t_feature_maps, 1, filter_h, t_filter_w))  
                  t_pool_sizes.append((img_h-filter_h+1, img_w-t_filter_w+1))
              
              layer0_outputs = []
              x = T.matrix('x')  
              y = T.ivector('y')
              y_target = T.ivector('y_target')
              W = theano.shared(value = U, name = "Words")
              
              t_conv_layers = []
              layer0_inputs = W[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1, x.shape[1],W.shape[1]))                                  
              
              for i in xrange(len(t_filter_hs)):
                 filter_shape = t_filter_shapes[i]
                 pool_shape = t_pool_sizes[i]
                 t_conv_layer = LeNetConvPoolLayer(rng, input=layer0_inputs, image_shape=(stat_length*batch_size, 1, img_h, img_w), 
                                                       filter_shape=filter_shape, poolsize = pool_shape, non_linear=conv_non_linear)
         #        t_conv_layer = LeNetConvPoolLayer_pre(rng,  input=layer0_inputs, image_shape=(stat_length*batch_size, 1, img_h, img_w), initial_w = init_t_conv[i*2], initial_b = init_t_conv[i*2+1],
         #                                              filter_shape=filter_shape, poolsize = pool_shape, non_linear=conv_non_linear, single_batch = stat_length)
                  
                 t_conv_layers.append(t_conv_layer)
                 layer0_outputs.append(t_conv_layer.output.flatten(2))
              
              layer1_inputs = T.concatenate(layer0_outputs, axis=1)
         
              
              layer1_inputs = layer1_inputs.reshape((batch_size,1,layer1_inputs.shape[0]/batch_size,layer1_inputs.shape[1]))     
              
              t_hidden_units[0] = t_feature_maps*len(t_filter_hs)
              
              
              s_img_h = stat_length
              s_img_w = t_hidden_units[0]
              s_filter_w = t_hidden_units[0]
              s_feature_maps = s_hidden_units[0]
              s_filter_shapes = []
              s_pool_sizes = []
              for filter_h in s_filter_hs:
                  s_filter_shapes.append((s_feature_maps, 1, filter_h, s_filter_w))
                  s_pool_sizes.append((s_img_h-filter_h+1, s_img_w-s_filter_w+1))
                  
              s_conv_layers = []
              layer1_outputs = []
              layer2_input = []
         
              for i in xrange(len(s_filter_hs)):
                  filter_shape = s_filter_shapes[i]
                  pool_shape = s_pool_sizes[i]
                  s_conv_layer = LeNetConvPoolLayer(rng, input=layer1_inputs, image_shape=(batch_size, 1, s_img_h, s_img_w),
                                                        filter_shape=filter_shape, poolsize = pool_shape, non_linear=conv_non_linear)
         #         s_conv_layer = LeNetConvPoolLayer_pre(rng, input=layer1_inputs, image_shape=(batch_size, 1, s_img_h, s_img_w), initial_w = init_s_conv[i*2], initial_b = init_s_conv[i*2+1],
         #                                               filter_shape=filter_shape, poolsize = pool_shape, non_linear=conv_non_linear)
         
                  s_conv_layers.append(s_conv_layer)
          
                  layer2_input.append(s_conv_layer.output.flatten(2))
              
              s_hidden_units[0] = s_feature_maps*len(s_filter_hs) 
              
              
              img_w = 300     
              img_h_report = len(train_report[0])    
              filter_w = img_w  
              feature_maps = hidden_units[0]
              filter_shapes = []
              pool_sizes = []
              for filter_h in t_filter_hs:
                 filter_shapes.append((feature_maps, 1, filter_h, filter_w))
                 pool_sizes.append((img_h_report-filter_h+1, img_w-filter_w+1))
              
              xx = T.matrix('xx') 
              zero_vec_tensor = T.vector()
              zero_vec = np.zeros(img_w)
             
              layer0_input_report = W[T.cast(xx.flatten(),dtype="int32")].reshape((xx.shape[0],1,xx.shape[1],W.shape[1]))                                  
              
              conv_layers_report = []
              layer0_output = []
              for i in xrange(len(t_filter_hs)):
                 filter_shape = filter_shapes[i]
                 pool_size = pool_sizes[i]
                 conv_layer_report = LeNetConvPoolLayer(rng, input=layer0_input_report,image_shape=(batch_size, 1, img_h_report, img_w),
                                         filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
         #        conv_layer_report = LeNetConvPoolLayer_pre(rng, input=layer0_input_report,image_shape=(batch_size, 1, img_h_report, img_w), initial_w = init_r_conv[i*2], initial_b = init_r_conv[i*2+1],
         #                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
         
                 layer2_input.append(conv_layer_report.output.flatten(2))
                 conv_layers_report.append(conv_layer_report)        
                 
              layer2_inputs = T.concatenate(layer2_input, 1)    
              hidden_units[0] = feature_maps*( len(t_filter_hs) + len(t_filter_hs) )
              
              classifier = MLPDropout(rng, input=layer2_inputs, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
              classifier_target = MLPDropout(rng, input=layer2_inputs, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
               
              params = classifier.params
              target_params = classifier_target.params       
              conv_params = []
              for conv_layer in t_conv_layers:
                  params += conv_layer.params
                  conv_params += conv_layer.params
                  target_params  += conv_layer.params
              for conv_layer in s_conv_layers:
                  params += conv_layer.params
                  conv_params += conv_layer.params
                  target_params  += conv_layer.params
              for conv_layer in conv_layers_report:
                  params += conv_layer.params
                  conv_params += conv_layer.params
                  target_params += conv_layer.params
         
         
                  
              cost = classifier.negative_log_likelihood(y) 
              dropout_cost = classifier.dropout_negative_log_likelihood(y)           
              grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)
        
              cost_target = classifier_target.negative_log_likelihood(y) 
              dropout_cost_target = classifier_target.dropout_negative_log_likelihood(y)           
              grad_updates_target = sgd_updates_adadelta(target_params, dropout_cost_target, lr_decay, 1e-6, sqr_norm_lim)
        
               
              train_y_pred = classifier.predict(layer2_inputs)
              train_error = T.mean(T.neq(train_y_pred, y))
              
              train_y_pred_target = classifier_target.predict(layer2_inputs)
              train_error_target = T.mean(T.neq(train_y_pred_target, y))
           
              train_model = theano.function([x,xx,y], cost, updates=grad_updates,
                                                          allow_input_downcast=True) 
              train_target_model = theano.function([x,xx,y], cost_target, updates=grad_updates_target,
                                                          allow_input_downcast=True) 
              
              train_model_error = theano.function([x, xx, y], classifier.errors(y),
                                                          allow_input_downcast=True)
              train_target_model_error = theano.function([x, xx, y], classifier_target.errors(y),
                                                          allow_input_downcast=True)
              get_layer0_output = theano.function([x, xx], layer0_output,
                                                          allow_input_downcast=True, on_unused_input='ignore')
              get_layer2_input = theano.function([x, xx], layer2_inputs,
                                                          allow_input_downcast=True, on_unused_input='ignore')
              get_classifier_params = theano.function([x, xx], conv_params,
                                                          allow_input_downcast=True, on_unused_input='ignore')
              
              test_size = x.shape[0]
              test_layer0_outputs = []
              test_layer2_input = []
              test_layer0_inputs = W[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1, img_h,W.shape[1]))                                  
            
               
              for conv_layer in t_conv_layers:
                test_layer0_output = conv_layer.predict(test_layer0_inputs, img_shape=(test_c*stat_length, 1, img_h, img_w))
                test_layer0_outputs.append(test_layer0_output.flatten(2)) 
              test_layer1_inputs = T.concatenate(test_layer0_outputs, 1)
             
              test_layer1_inputs = test_layer1_inputs.reshape((test_c,1,test_layer1_inputs.shape[0]/test_c,test_layer1_inputs.shape[1]))     
             
              test_layer1_outputs = []
              for conv_layer in s_conv_layers:
                 test_layer1_output = conv_layer.predict(test_layer1_inputs,img_shape=(test_c, 1, s_img_h, s_img_w))
                 test_layer2_input.append(test_layer1_output.flatten(2)) 
             
             
             
              test_size_report = xx.shape[0]
              test_layer0_outputs = []
              test_layer0_inputs_report = W[T.cast(xx.flatten(),dtype="int32")].reshape((xx.shape[0],1, img_h_report,W.shape[1]))                                  
              for conv_layer in conv_layers_report:
                 test_layer0_output = conv_layer.predict(test_layer0_inputs_report,img_shape=(test_c, 1, img_h_report, img_w))
                 test_layer2_input.append(test_layer0_output.flatten(2)) 
             
              test_layer2_inputs = T.concatenate(test_layer2_input, 1)
              test_y_pred = classifier_target.predict(test_layer2_inputs)
              test_y_pred_p = classifier_target.predict_p(test_layer2_inputs)
            
              test_model_test_y_pred_p = theano.function([x, xx], test_y_pred_p,
                                                      allow_input_downcast=True)
              test_model_get_test_layer2_inputs = theano.function([x, xx], test_layer2_inputs,
                                                      allow_input_downcast=True, on_unused_input='ignore')
              test_model_test_y_pred = theano.function([x, xx], test_y_pred,
                                                      allow_input_downcast=True)  
           
         
              train_labels = train_labels.reshape(len(train_labels))
              target_labels = target_labels.reshape(len(target_labels)) 
              
              n_batches = int(len(train_code)/batch_size)
              n_train_batches = n_batches
              n_batches_target = int(len(target_code)/batch_size)
              n_train_batches_target = n_batches_target
        #      n_train_batches = int(np.round(n_batches*0.9))
              n_val_batches = n_batches-n_train_batches
              best_val = 0        
              test_acc = 0 
         
              val_error = []
              train_code_x = []
              train_code_y = [] 
              target_code_x = []
              target_code_y = [] 
              classifier_params = []
              test_acc_all = []
              print "start training..."
              for epoch in xrange(n_epochs):
                   start_time = time.time()
                   train_error, train_error_target = [], []
         
                   for index_i in xrange(n_train_batches):
                       train_code_x, train_report_x, train_y = get_batch(train_code=train_code, train_report = train_report, train_label=train_labels, index_i=index_i, batch_size=batch_size)             
        
                       cost = train_model(train_code_x, train_report_x, train_y) 
                       train_error.append(train_model_error(train_code_x, train_report_x,train_y))   
                   
                   for index_i in xrange(n_train_batches_target):
                       
                       target_code_x, target_report_x, target_y = get_batch(train_code=target_code, train_report = target_report, train_label=target_labels, index_i=index_i, batch_size=batch_size)             
        
                       cost_target = train_target_model(target_code_x, target_report_x, target_y) 
                       train_error_target.append(train_target_model_error(target_code_x, target_report_x, target_y))  
                       
                   train_perf = 1 - np.mean(train_error)
                   train_target_perf = 1 - np.mean(train_error_target)
                   print('epoch: %i, training time: %.2f secs,  train perf: %.2f %%, train target perf: %.2f %%' % (epoch, time.time()-start_time,  train_perf * 100., train_target_perf * 100. ) )  
     
                   
              print "Finish Training. Begin Testing!" 
              test_time = time.time()
              top1_single = []
              top3_single = []
              top5_single = []
              top10_single = []
              top20_single = []
              acc_single = []
              auc_single = []
              map_single = []
              mrr_single = []
              for test_index in xrange(test_num):
                   print str(test_index),
                   X = cPickle.load(open(project_name + "_"+target_name+"/transfer/"+str(repeat_index)+"/middle/test_middle"+str(test_index)+".csv","rb"))
                   test_report, test_code, test_labels = X[0], X[1], X[2]
                   test_report = np.array(test_report,dtype="int") 
                   test_labels = np.array(test_labels,dtype="int")
                   test_code_x, test_report_x, test_label = get_testsets(test_code, test_report, test_labels)             
    
    
                   test_p = []
                   test_y_predict = test_model_test_y_pred(test_code_x, test_report_x)
                   test_y_pred_p = test_model_test_y_pred_p(test_code_x, test_report_x)
                   test_p = test_y_pred_p[:,1]
                   test_y_labels = test_labels
    
                   
                   print_results(test_p, test_y_predict, test_y_labels, test_index, project_name, target_name)
                   acc, top1, top3, top5, top10, top20, auc, mapp, mrr = eval_y(test_p, test_y_predict, test_y_labels)
                   top1_single.append(top1)
                   top3_single.append(top3)
                   top5_single.append(top5)
                   top10_single.append(top10)
                   top20_single.append(top20)
                   acc_single.append(acc)
                   auc_single.append(auc)
                   map_single.append(mapp)
                   mrr_single.append(mrr)
                   
                   top1_all.append(top1)
                   top3_all.append(top3)
                   top5_all.append(top5)
                   top10_all.append(top10)
                   top20_all.append(top20)
                   acc_all.append(acc)
                   auc_all.append(auc)
                   map_all.append(mapp)
                   mrr_all.append(mrr)
              print "\n", np.mean(top1_single), np.mean(top3_single), np.mean(top5_single), np.mean(top10_single), np.mean(top20_single), np.mean(auc_single), np.mean(map_single), np.mean(mrr_single)
              write_f = file( project_name + "_"+target_name+"/transfer/"+str(repeat_index) + "/summary.out",'w+')
              write_f.write(str(np.mean(top1_single)) + ", " + str(np.mean(top3_single)) + ", " + str(np.mean(top5_single)) + ", " + str(np.mean(top10_single)) + ", " + str(np.mean(top20_single)) + ", " +str(np.mean(auc_single))+ ", " + str(np.mean(map_single)) + ", " + str(np.mean(mrr_single))+ "\n")
              write_f.close()
             
              print "Finish Testing!"                    
              print('test time: %.2f secs' % (time.time()-test_time) )  
            
           write_time = file(project_name + "_" +target_name + "/transfer/time.txt","w+")
           write_time.write('project time: %.2f secs' % (time.time()-project_time) )
           write_time.close()
                
           print "\n Summary: \n", np.mean(top1_all), np.mean(top3_all), np.mean(top5_all), np.mean(top10_all), np.mean(top20_all), np.mean(auc_all), np.mean(map_all), np.mean(mrr_all)  

           top1_task.append(np.mean(top1_all))
           top3_task.append(np.mean(top3_all))
           top5_task.append(np.mean(top5_all))
           top10_task.append(np.mean(top10_all))
           top20_task.append(np.mean(top20_all))
           acc_task.append(np.mean(acc_all))
           auc_task.append(np.mean(auc_all))
           map_task.append(np.mean(map_all))
           mrr_task.append(np.mean(mrr_all))
           print('all time: %.2f secs' % (time.time()-all_time) ) 

       print "Done!"
     
