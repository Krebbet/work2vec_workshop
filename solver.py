'''
This solver unit is to simplify the model 
code and to reduce repeated code...

it is designed to take in a model and data and train the thing.
'''

import os
import numpy as np
import math
import io
import tensorflow as tf



LOG_DIR = 'logs'


class Solver(object):
  """
   This creates a draw model architecture to be trained on Tensor flow
   A = input image x size
   B = input image Y size
   enc_size = Encoder LSTM size
   dec_sixe = Decoder LSTM size
   read_n = Read Filter patch size
   write_n = write filter patch size
   
   h_pred_size = size of hidden layer in classification step after
      the attention model has done its job and produced a z vector.
   
  """
  
  def __init__(self,model):
    self.m = model


   
      
  def optimize(self,learning_rate,var_list = None):    
    ## OPTIMIZER ## 
    
    model = self.m
    with tf.name_scope('Optimize'):
      # the adam optimizer is much more stable and optimizes quicker than
      # basic gradient descent. (*Ask me for details*)
      optimizer=tf.train.AdamOptimizer(learning_rate, beta1=0.5)
      
      # This is generally for fine-tuning,
      # Instead of optimizing all variables you can 
      # choose a set to train.
      if (var_list == None):
        grads=optimizer.compute_gradients(model.loss)
      else:
        grads=optimizer.compute_gradients(model.loss,var_list = var_list)
        
      # we cap the gradient sizes to make sure we do not
      # get outlier large grads from computational small number errors
      with tf.name_scope('Clip_Grads'):
        for i,(g,v) in enumerate(grads):
            if g is not None:
                grads[i]=(tf.clip_by_norm(g,5),v) # clip gradients
                
        self.train_op = optimizer.apply_gradients(grads)
          
    return         

  def train(self,target_words,context,param):
    # define params locally for readability
    model = self.m     
    learning_rate = param['learning_rate'] 
    epoch = param['epoch']
    batch_size = param['batch_size']
    id = param['id']
    num_skips =param['num_skips']
    skip_window = param['skip_window']
    
    # set up all the directories to send the model and logs...
    model_dest = os.path.join(model.name, id)
    log_dir = os.path.join(model_dest, LOG_DIR)
    #train_dir = os.path.join(log_dir, 'train')
    #test_dir = os.path.join(log_dir, 'test')    
    
    '''
      ckpt_dir = 'model',
      ckpt_name = 'trained_model',
      log_dir = 'logs',
      read_out = 100
    '''
    print('***********************************************')
    print('Begining training of %s model is id: %s' % (model.name,id))
  
    # get the model directory and make sure it exists
    # ckpt = utils.get_model_dir(ckpt_name,ckpt_dir)    
  
  
    # get training op.. set learning rate...
    self.optimize(learning_rate)

    # define some initial vars...



    # set the training feed.
    fetches=[]
    fetches.extend([self.train_op])

    # determine the counter values.
    num_train = len(context_words)
 
    iterations_per_epoch = max(num_train // batch_size, 1)
    if (num_train % batch_size) != 0: 
      iterations_per_epoch -= 1
    num_iterations = epoch * iterations_per_epoch
    
    
    print('Training Points:',num_train)
    print('Batch size =',batch_size)
    print('Epoch:',epoch,'(iters/epoch):',iterations_per_epoch)
    print('Total Iterations:',num_iterations)
    print('model will be saved to: %s' % model_dest)
    print('logs will be stored in: %s' % log_dir)
    
    
    with tf.Session() as sess:

      # initialize variables -> need to add code to load if loading trained model.
      tf.global_variables_initializer().run()
  
      # create session writers
      writer = tf.summary.FileWriter(log_dir,sess.graph) # for 1.0
      #test_writer = tf.summary.FileWriter(os.path.join(test_dir , model.model_name)) # for 1.0
      merged = tf.summary.merge_all()
      
      
      # setup a saver object to save model...
      # create model saver
      saver = tf.train.Saver()
      
      print('Begin Training')
      print('***********************************************')
      
      for e in range(epoch):
        # create a mask to shuffle the data
        mask = np.arange(num_train)
        np.random.shuffle(mask)

        for i in range(iterations_per_epoch):
          if (i % param['read_out'] == 0):
            print('%d of %d for epoch %d' %(i,iterations_per_epoch,e))
          # Grab the batch data...
          target_batch = target_words[mask[batch_size*i:batch_size*(i +1)]]
          context_batch = context_batch[mask[batch_size*i:batch_size*(i +1)]]
          

          feed_dict={model.target_words:target_batch,
                     model.context:context_batch,
                     model.is_training:True}      
          

          # do training on batch, return the summary and any results...
          [summary,_]=sess.run([merged,fetches],feed_dict)

          
          # write summary to writer
          writer.add_summary(summary, i + e*iterations_per_epoch)
        
        
        # epoch done, check word similarities....
        # Note: I do not have access to the word library so 
        # I cannot create my own reverse lookup...
        if (e % param['similarity_readout'] == 0):
          [sim] = sess.run([model.similarity])
          for i in range(model.test_size):
            #valid_word = reverse_dictionary[model.valid_examples[i]]
            valid_word = model.valid_examples[i]
            top_k = 8  # number of nearest neighbors
            #x = -sim[i, :].argsort()
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
              #close_word = reverse_dictionary[nearest[k]]
              close_word = nearest[k]
              #log_str = '%s %s,' % (log_str, close_word)
              log_str = '%s %d,' % (log_str, close_word)
            print(log_str)        
      
        
        #[train_acc]=sess.run([model.accuracy],feed_dict)
        #acc = self.test_acc(sess,data['X_test'],data['y_test'],test_writer,batch_size,iter = (e+1)*iterations_per_epoch)
        if (e % param['check_point'] == 0):
          saver.save(sess,model_dest, global_step=e+1)
        print('%d of %d epoch complete.' % (1+e,epoch))
        
        
  
      # saves variables learned during training
      ## TRAINING FINISHED ##
      saver.save(sess,model_dest)  
      
      #test_writer.close()
      writer.close()
      sess.close()  


    print('***********************************************')
    print('Done training')
    print('model saved to: %s' % model_dest)
    print('logs stored in: %s' % log_dir)
    print('***********************************************')
    return