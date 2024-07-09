# main.py
from preprocess import preprocess
from inital_preprocessing import inital_preprocessing
from model import EfficientNetB0
from logger import setup_logging
import os
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch.optim as optim

logger = setup_logging('application.log') # setting up logger
dataset_dir = r"E:\Projets\face\FMD_DATASET" #Dataset path

if os.path.exists(dataset_dir + "_Split"): # Checking if the dataset is already split then skipping the split 
    logger.info('DataSet already Split')    
else:
    i = inital_preprocessing(dataset_dir)
    logger.info("Initial preprocessing done")
    
new_input = dataset_dir + "_Split"
face_folder = os.path.dirname(dataset_dir)

if os.path.exists(os.path.join(face_folder , 'train')): # checking if dataset is already processed
    logger.info("Data already processed")
    req = False # no need for preprocessing
else:
    logger.info("Preprocessing the Data")
    req = True # Need to preprocess

pre = preprocess(new_input, face_folder , 32, req)
logger.info('DataLoaders Ready')
loaders = pre.DataLoaders # dataLoaders ready


#define model
model = EfficientNetB0(num_classes=2)  # Change num_classes according to your dataset
model.training(train_loader=loaders[0] , val_loader=loaders[1] ,
                epochs=10 , output_path=face_folder)





