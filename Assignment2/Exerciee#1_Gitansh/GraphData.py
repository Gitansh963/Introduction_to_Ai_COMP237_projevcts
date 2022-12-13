'''
@author: Gitansh Mittal

Reference: https://www.python.org/doc/essays/connectionss/
'''

#create a dictionary with all the mappings
# connections = {}
# connections["Gitansh"] = {"George", "Frank", "Adam"}
# connections["Frank"] = {"Gitansh"}
# connections["George"] = {"Gitansh"}
# connections["Adam"] = {"Ema", "Gitansh", "Bob"}
# connections["Ema"] = {"Dolly", "Bob", "Adam"}
# connections["Bob"] = {"Adam", "Dolly", "Ema"}
# connections["Dolly"] = {"Ema", "Bob"}




graph = {'Adam':['Bob','Gitansh','Ema'],
         'Bob':['Adam','Dolly','Ema'],
         'Dolly':['Ema','Bob'],
         'Ema':['Bob','Adam','Dolly'],
         'Gitansh':['Adam','Frank','George'],
         'George':['Gitansh'],
         'Frank':['Gitansh']
         }