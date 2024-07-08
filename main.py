# main.py
from preprocess import preprocess
from inital_preprocessing import inital_preprocessing
from logger import setup_logging
import os

logger = setup_logging('application.log')


dataset_dir = r"E:\Projets\face\FMD_DATASET"
i = inital_preprocessing(dataset_dir)
logger.info("Initial preprocessing done.")


face_folder = os.path.dirname(dataset_dir)
pre = preprocess(i.output_path, face_folder , 32)
logger.info('DataLoaders created')
loaders = pre.DataLoaders





