import xmlrpc.client

#configure the absolute model path
modelPath = "D:\design_automation\plecs"
#configure the XML-RPC port -> needs to coincide with the PLECS configuration
port = "1080"
   
# start PLECS
server = xmlrpc.client.Server("http://localhost:" + port + "/RPC2")

################################################################################
# Example: Parameter sweep of a boost converter
################################################################################

# define model name, scope path and resistance paths
modelName = "Boost"
scopeRef = modelName+'/Scope'
resistorPath = modelName+'/R1'

# open the model using the XMLRPC server, needs absolute path
server.plecs.load(modelPath+'\Boost.plecs')
#clear existing traces in the scope
server.plecs.scope(scopeRef,'ClearTraces')

opts = { 'ModelVars' : {'varR' : 4} }

# define resistor values for parameter sweep
RVals = [3, 4, 5]

# loop for all values
for i in RVals:
  #set value for R1
  opts['ModelVars']['varR'] = i
  #start simulation, using opts struct
  server.plecs.simulate(modelName, opts)
  #add trace to the scope
  server.plecs.scope(scopeRef, 'HoldTrace', 'R = {0} ohms'.format(i))
