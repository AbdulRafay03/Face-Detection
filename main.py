# main.py

from inital_preprocessing import inital_preprocessing
from logger import setup_logging

logger = setup_logging('application.log')

dataset_dir = r"E:\Projets\face\FMD_DATASET"
i = inital_preprocessing(dataset_dir)
output_path = dataset_dir + "_Split"
i.split(output_path)
logger.info("Initial preprocessing done.")

