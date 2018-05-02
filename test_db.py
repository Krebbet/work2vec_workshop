
import MySQLdb as mysql
import yaml
import numpy as np

PARAM_FILE_DIRECTORY = 'parameters.yml'    

# load parameters for run...
parameters = yaml.load(open(PARAM_FILE_DIRECTORY))
db_defs = parameters['database']
solver_param = parameters['solver_param']        
model_param = parameters['model_param']













print('trying to connect to db...')
# create connection to the database.
db = mysql.connect(host=db_defs['ip'],
                  user=db_defs['usrid'],
                  passwd=db_defs['password'],
                  db=db_defs['name'])


print('connected to db...')  


def mysql_query(cmd,maxrows=0):
  db.query(cmd)
  r=db.store_result()
  return r.fetch_row(maxrows=maxrows)



#db.query("""SHOW DATABASES""")
str = ("""USE %s """ % 'greenwichhr_w2v_workbench')
print(str)
db.query(str)
db.query("""SHOW tables""")
r=db.store_result()
print(r.fetch_row(maxrows=0))

db.query("""DESCRIBE _test_input_pair_list""")

r=db.store_result()
print(r.fetch_row(maxrows=0))

res = mysql_query("""SELECT COUNT(*) FROM _test_input_pair_list""")
data_length = res[0][0]


#res = mysql_query("""SELECT var_1,var_2 FROM _test_input_pair_list  LIMIT 55,10""")
#print(res)



#data = mysql_query("""SELECT var_1,var_2 FROM _test_input_pair_list LIMIT 55,10""")
data = mysql_query("""SELECT var_1,var_2 FROM _test_input_pair_list LIMIT 55,10""")
data = np.asarray(data)
#print(data)

number_of_samples = None
if number_of_samples != None:
  str = ("""SELECT var_1,var_2 FROM _test_input_pair_list LIMIT %d""" % number_of_samples)
else:
  str = ("""SELECT var_1,var_2 FROM _test_input_pair_list""")
 
         
data = mysql_query(str)
         
print('XXXXXXXXXXXXXXXXXXXXXXXXXxx')
data = np.asarray(data)
print(data.shape)
x = data[:,0]
print(x.shape)
print(x.shape[0])
         
         
         
         
         
         
         

