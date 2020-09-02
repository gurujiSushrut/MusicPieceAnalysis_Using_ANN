'''
Created on Sep 2, 2020

@author: admin
'''
import os
from AudioPreprocessing.FouriorTransform import FouriorTransform
import json

class Preprecessor(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        self.FT = FouriorTransform()
        
    def save_mfcc(self,dataset_path,json_path,sample_rate = 20000,n_mfcc = 13, n_fft=2048,hop_length=512,num_segments = 5):
        data = {
            "mapping" :[],
            "mfcc":[],
            "labels":[]
            }
        for i,(dirpath,dirnames,filenames) in enumerate(os.walk(dataset_path)):
            if dirpath is not dataset_path:
                data["mapping"].append(dirpath.split(os.path.pathsep)[-1])
                for f in filenames:
                    file_path = os.path.join(dirpath,f)
                    if num_segments >1:
                        mfcc_list = self.FT.MFCC(file_path, n_fft = n_fft, hop_length=hop_length, n_mfcc=n_mfcc,sample_slice = True, num_segments = num_segments)
                    else:
                        mfcc_list = self.FT.MFCC(file_path, n_fft = n_fft, hop_length=hop_length, n_mfcc=n_mfcc,sample_slice = False, num_segments = num_segments)
                    for mfcc in mfcc_list:
                        mfcc = mfcc.T
                        num, exp = self.FT.get_Expected_Vectors_Per_Segment(self.FT.get_signal_vector(file_path,sample_rate), sample_rate, num_segments, hop_length)
                        if len(mfcc)==exp:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i-1)
        with open(json_path,"w") as fp:
            json.dump(data,fp, indent = 4)
        
                        
                        
                    
                    
                