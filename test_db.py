
import MySQLdb as mysql
import yaml

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