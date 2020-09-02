'''
Created on Sep 2, 2020

@author: admin
'''
import json

class Perceptron(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
    def load_data(self,dataset_path):
        with open(dataset_path,"r") as fp:
            data = json.load(fp)
        