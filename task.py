#import MySQLdb as mysql
import yaml


# local imports... -> load the 
from solver import Solver
from utils import create_model 
from utils import collect_data
import numpy as np

from models.constants import *


# extract all parameters for the yml definitions file
PARAM_FILE_DIRECTORY = 'parameters.yml'    

def main():

  # load parameters for run...
  parameters = yaml.load(open(PARAM_FILE_DIRECTORY))
  db_defs = parameters['database']
  solver_param = parameters['solver_param']        
  model_param = parameters['model_param']
  model_specific_params = parameters[model_param['model_type']]

  
  

  print('Collect Data ....')
  data, count, dictionary, reverse_dictionary = collect_data(vocabulary_size=model_param['vocabulary_size'])
  print('Done Collect Data.')
  
  #skip_window = 2       # How many words to consider left and right.
  #num_skips = 2         # How many times to reuse an input to generate a label.

  # We pick a random validation set to sample nearest neighbors. Here we limit the
  # validation samples to the words that have a low numeric ID, which by
  # construction are also the most frequent.
  num_sampled = 64    # Number of negative examples to sample.
  
  
  '''
  print('trying to connect to db...')
  # create connection to the database.
  db = mysql.connect(host=db_defs['ip'],
                    user=db_defs['usrid'],
                    passwd=db_defs['password'],
                    db=db_defs['name'])
  
  
  print('connected to db...')  
  '''

  '''
  # setup params for training
  ckpt_file=os.path.join(options.dir,options.model_name)
  log_dir=os.path.join(options.logdir,'logs')
  print(ckpt_file)
  print(log_dir)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)  
  '''
  print('try to import model')
  # build model...
  model = create_model(model_param,model_specific_params)
  print('Model drawn')
                            
             

  # Initialize the solver object.
  solver = Solver(model)
  
  # train model....
  solver.train(data,dictionary,reverse_dictionary,solver_param)

  
  print('done!')
  
  
  # do a last validation???

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    


                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  