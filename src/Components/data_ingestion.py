from abc import ABC, abstractmethod
from logger import logging
import pandas as pd
import zipfile
import os

# Define ab abstract class for Data Ingestion 
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path:str) -> pd.DataFrame:
        """
        Abstract method to ingest data from a given file. 
        """
        pass


# Impleted another class for ZIP Ingestion 
class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path:str) -> pd.DataFrame:
        """
        Extracts a zip file and returns the contect as a pandas Dataframe. 
        """

        # Ensure the file is .zip file
        if not file_path.endswith(".zip"):
            logging.info("Error: File is not .zip file ")
            raise ValueError("The provided file is not a zip file ")
        
        # Extract the zip file ( it's reference is stored in zip_ref)
        with zipfile.ZipFile(file_path,"r") as zip_ref:
            zip_ref.extractall("extracted_data")

        
        # Find the extracted CSV file ( assuming there is one CSV file inside the zip)
        extracted_files=os.listdir("extracted_data")
        csv_files=[ file for file in extracted_files if file.endswith(".csv")]

        if len(csv_files) == 0:
            logging.info("No CSV files in the extracted data --> zip")
            raise FileNotFoundError("No CSV file is found in the extracted data")
        
        if len(csv_files) > 1:
            logging.info("More than one files.")
            raise ValueError("Multiple CSV files found. Please specify which one to use.")
        
        # Read the CSV into a DataFrame
        csv_file_path= os.path("extracted_data",csv_files[0])
        df=pd.read_csv(csv_file_path)
        logging.info("DataFrame Created")

        # Return the DataFrame 
        return df


# Implement a factory to create DataIngestors
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension:str) -> DataIngestor:
        if file_extension == ".zip":
            logging.info("Zip_data_ingestor is called..")
            return ZipDataIngestor()
        else:
            logging.info("Zip data ingestor is not available here.")
            raise ValueError(f"No ingestor available for file extention: {file_extension}")


if __name__ == "__main__":
    # Proving the file path
    file_path = "C:\Users\Mayank kumar\Desktop\End-to_End House Prediction Model\data\archive.zip"

    # Extracting the file extention 
    file_extention=os.path.splitext(file_path)[1]




