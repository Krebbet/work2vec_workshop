
import MySQLdb as mysql
import yaml

import os
import numpy as np
import importlib
import urllib


def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError # evil ValueError that doesn't tell you what the wrong value wa
         

         
         
def mysql_query(cmd,maxrows=0):
  db.query(cmd)
  r=db.store_result()
  return r.fetch_row(maxrows=maxrows)

         
         
def connect_to_db(db_defs):
  print('trying to connect to db...')
  # create connection to the database.
  db = mysql.connect(host=db_defs['ip'],
                    user=db_defs['usrid'],
                    passwd=db_defs['password'],
                    db=db_defs['name'])
         
  return db                  



         
def collect_data(db_defs,number_of_samples = None):
    # connect to db 
    db = connect_to_db(db_defs)
    
    #choose db....
    str = ("""USE %s """ % db_defs['name'])
    db.query(str)

    # get the data count from the db
    #res = mysql_query("""SELECT COUNT(*) FROM _test_input_pair_list""")
    #count = res[0][0]
    
    if number_of_samples != None:
      str = ("""SELECT var_1,var_2 FROM _test_input_pair_list LIMIT %d""" % number_of_samples)
    else:
      str = ("""SELECT var_1,var_2 FROM _test_input_pair_list""")
    # grab the data...
    data = mysql_query(str)
    data = np.asarray(data)
    target_words = data[:,0]
    context = data[:,1]

    return target_words, context

    
    
def create_model(model_param,model_specific_params):
    import_string = ('%s' % model_param['model_type'])
    model_def = importlib.import_module(import_string)
    return model_def.Model(model_param,model_specific_params)

      