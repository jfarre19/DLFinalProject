from tensorflow.keras.utils import Sequence
from math import floor
import numpy as np

# Keras Sequencer allows us to avoid loading all image data at once and supports batching
# Google QuickDraw dataset is about 38 GB
class DataGenerator(Sequence):
    def __init__(self, metadata, batch_size):
        self.metadata = metadata
        self.batch_size = batch_size
        
        self.n_classes = self.metadata.shape[0]
        
        self.n_images = sum(self.metadata['Image Count'])
        
        self.indexes = np.zeros(self.n_images, dtype=np.int64)
        self.classes = np.zeros(self.n_images, dtype=np.int64)
        start = 0
        for idx,row in self.metadata.iterrows():
            count = row['Image Count']
            
            self.indexes[start:start+count] = np.arange(count)
            self.classes[start:start+count] = idx
            
            start += count
            
        order = np.arange(self.n_images, dtype=np.int64)
        np.random.shuffle(order)
        
        self.indexes = self.indexes[order]
        self.classes = self.classes[order]
        
        self.data = []
        for file in self.metadata['File Name']:
            self.data.append(np.load(file, mmap_mode='r'))
        
        self.size = self.data[0].shape[1]
            
        
    def __len__(self):
        return floor(self.n_images / self.batch_size)
    
    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx+1) * self.batch_size
        
        indexes = self.indexes[start:end]
        classes = self.classes[start:end]
        
        X = np.zeros((self.batch_size,self.size))
        y = classes
        
        for idx in range(self.batch_size):
            X[idx,:] = self.data[classes[idx]][indexes[idx],:]
        
        X = X.reshape(self.batch_size,28,28)
        return X,y