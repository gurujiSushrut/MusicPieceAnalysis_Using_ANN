'''
Created on Aug 29, 2020

@author: admin
'''
import math
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment as AS 
import os
from playsound import playsound


class FouriorTransform(object):
    
    def __init__(self):
        '''
        Constructor
        '''
        self.Amplitude = 0
        self.Frequency = 0
        self.Phy = 0
        self.plots = 0
        self.subplots = 0
    
    def desectSoundWaves(self):
        pass
    
    def attachSoundWaves(self,Wave1,Wave2):
        f1= Wave1["frequency"] 
        A1= Wave1["amplitute"]
        ph1=Wave1["phy"]
        
        f2= Wave2["frequency"] 
        A2= Wave2["amplitute"]
        ph2=Wave2["phy"]   
        
        t=ph1-ph2
        
        s= A1*math.sin(2*math.pi*f1*t+ph1)+A2*math.sin(2*math.pi*f2*t+ph2)
        
        return s
    
    def convertAudioFileToWAV(self,file):
        if os.path.isfile(file):
            filename, fileext = os.path.splitext(file)
            if ".mid" in fileext.casefold():
                print("Cannot Convert MIDI files. Please convert your file here: 'https://www.zamzar.com/convert/midi-to-wav/'")
                return
            else:
                print("File Exists")
                filename, fileext = os.path.splitext(file)
                if fileext != ".wav":
                    audioFile = AS.from_file(file, format = fileext.split(".")[-1])
                    audioFile.export(filename+".wav",format = "wav")
                return (filename+".wav")
        else:
            print("File Dosn't exist")
    
    def getWavePlot(self,file,sr = 20000):
        wav = self.convertAudioFileToWAV(file)
        if wav != None:
            if self.subplots>=4 and self.subplots>=4*self.plots:
                self.new_figure()
    
            plt.subplot(2,2,(self.subplots%4)+1)
            signal, sr = librosa.load(wav, sr=sr)
            librosa.display.waveplot(signal,sr=sr)
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            self.subplots +=1
    
        
    def FFT(self,file, half_transform=True, sample_rate = 20000):
        wav = self.convertAudioFileToWAV(file)
        if wav != None:
            signal, sr = librosa.load(wav, sr=sample_rate)
            
            fft = np.fft.fft(signal)
            magnitude = np.abs(fft)
            frequency = np.linspace(0,sr,len(magnitude))
            
            if half_transform:
                left_magnitude = magnitude[:int(len(frequency)/2)]
                left_frequency = frequency[:int(len(frequency)/2)]
                return(left_frequency, left_magnitude)
        
        return(frequency, magnitude)
    
    def STFT(self,file,log=True, sample_rate = 20000,n_fft = 2048,hop_length = 512):
        wav = self.convertAudioFileToWAV(file)
        stft = None
        if wav != None:
            signal, sr = librosa.load(wav, sr=sample_rate)
            stft  = librosa.core.stft(signal,n_fft = n_fft,hop_length = hop_length)
            if log:
                spec= librosa.amplitude_to_db(np.abs(stft))
            else:
                spec = np.abs(stft)
        return spec
    def get_signal_vector(self,file,sample_rate = 20000):
        wav = self.convertAudioFileToWAV(file)
        signal, sr = librosa.load(wav, sr=sample_rate)
        return signal
    def get_Expected_Vectors_Per_Segment(self,signal,sample_rate = 20000,num_segments = 1,hop_length=512):
        DURATION = librosa.get_duration(y=signal, sr=sample_rate)
        num_samples_per_segment = (sample_rate*DURATION)/num_segments
        exptected_vectors_per_segment = math.ceil(num_samples_per_segment/hop_length)
        return num_samples_per_segment,exptected_vectors_per_segment
    def MFCC(self,file,n_fft = 2048,hop_length = 512,sample_rate = 20000,n_mfcc = 13,sample_slice = False, num_segments = 1):
        wav = self.convertAudioFileToWAV(file)
        mfcc = None
        mfcc_list = []
        if wav != None:
            signal, sr = librosa.load(wav, sr=sample_rate)
            if sample_slice == True:
                num_samples_per_segment,exptected_vectors_per_segment = self.get_Expected_Vectors_Per_Segment(signal, sample_rate=sample_rate, num_segments=num_segments, hop_length=hop_length)
                for s in num_segments:
                    start_sample = num_samples_per_segment*s
                    finish_sample = start_sample+num_samples_per_segment
                    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],n_fft=n_fft,hop_length=hop_length,n_mfcc=n_mfcc)
                    mfcc_list.append(mfcc)
            else:       
                mfcc = librosa.feature.mfcc(signal,n_fft=n_fft,hop_length=hop_length,n_mfcc=n_mfcc)
                mfcc_list.append(mfcc)
        return mfcc_list
        
    def ShowSpectogram(self,spectogram,hop_length = 512,sample_rate=20000,Xlabel="X axis",Ylabel="Y axis"):
        
        if self.subplots>=4 and self.subplots>=4*self.plots:
            self.new_figure()

        plt.subplot(2,2,(self.subplots%4)+1)
        librosa.display.specshow(spectogram,hop_length=hop_length,sr=sample_rate)
        plt.xlabel(Xlabel)
        plt.ylabel(Ylabel)
        plt.colorbar()
        self.subplots +=1
            
    def plot2DGraph(self,Xaxis,Yaxis,Xlabel="X axis",Ylabel="Y axis"):
         
        if self.subplots>4 and self.subplots>=4*self.plots:
            self.new_figure()
        plt.subplot(2,2,(self.subplots%4)+1)
        plt.plot(Xaxis,Yaxis)
        plt.xlabel(Xlabel)
        plt.ylabel(Ylabel)
        self.subplots +=1 
    
    def plot3DGraph(self,Xaxis,Yaxis,Zaxis,Xlabel="X axis",Ylabel="Y axis",Zlabel ="Z axis"):
        
        if self.subplots>4 and self.subplots>=4*self.plots:
            self.new_figure()
        plt.subplot(2,2,(self.subplots%4)+1)
        ax = plt.axes(projection='3d')
        ax.plot3D(Xaxis,Yaxis,Zaxis)
        ax.set_xlabel(Xlabel)
        ax.set_zlabel(Zlabel)
        ax.set_ylabel(Ylabel)
        self.subplots +=1
    
    def show_figures(self):
        plt.show()
        self.subplots = 0
    
    def new_figure(self):
        self.plots+=1
        plt.figure(self.plots)
        
    
    def play_Audio(self,file):
        try:
            if os.path.isfile(file):
                print("File Exists")
                playsound(file)
            else:
                print("File Doesnt exist")
             
        except Exception as e:
            print(e)
        
        
        
    

if __name__=="__main__":
    FT = FouriorTransform()
    
    Wave1 = {"frequency":8,"amplitute":6,"phy":2}
    Wave2 = {"frequency":13,"amplitute":89,"phy":77}
    print(FT.attachSoundWaves(Wave1, Wave2))
    FT.new_figure()
    #FT.play_Audio(r"D:\Music Files\17562__jo130__a-second-behind-picked-guitar-1-rec-16.wav")
    FT.getWavePlot(r"D:\Music Files\17562__jo130__a-second-behind-picked-guitar-1-rec-16.wav")
    frequency, magnitude = FT.FFT(r"D:\Music Files\17562__jo130__a-second-behind-picked-guitar-1-rec-16.wav")
    FT.plot2DGraph(frequency, magnitude, "Frequency", "Magnitude")
    
    stft = FT.STFT(r"D:\Music Files\17562__jo130__a-second-behind-picked-guitar-1-rec-16.wav")
    FT.ShowSpectogram(stft,Xlabel="Time",Ylabel="Frequency")
    
    mfcc = FT.MFCC(r"D:\Music Files\17562__jo130__a-second-behind-picked-guitar-1-rec-16.wav")
    FT.ShowSpectogram(mfcc,Xlabel="Time",Ylabel="Frequency")
    
    #FT.play_Audio(r"D:\Music Files\23977__connum__newswav-plusdrums.wav")
    FT.getWavePlot(r"D:\Music Files\23977__connum__newswav-plusdrums.wav")
    frequency, magnitude = FT.FFT(r"D:\Music Files\23977__connum__newswav-plusdrums.wav")
    FT.plot2DGraph(frequency, magnitude, "Frequency", "Magnitude")
    
    stft = FT.STFT(r"D:\Music Files\23977__connum__newswav-plusdrums.wav")
    FT.ShowSpectogram(stft,Xlabel="Time",Ylabel="Frequency")
    
    mfcc = FT.MFCC(r"D:\Music Files\23977__connum__newswav-plusdrums.wav")
    FT.ShowSpectogram(mfcc,Xlabel="Time",Ylabel="Frequency")
    
    FT.show_figures()
    
    
        
        