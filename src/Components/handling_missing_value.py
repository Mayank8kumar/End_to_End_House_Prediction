from logger import logging
from abc import ABC, abstractmethod
import pandas as pd


# Abstract Base Class for Handling Missing Values Strategy
class MissingValueHandlingStrategy(ABC):
    @abstractmethod
    def handle(self,df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method for handling missing value in the DataFrame.

        Parameters:
        df ( pd.DataFrame): The input DataFrame containing the missing values. 

        Returns:
        pd.DataFrame: The DataFrame with missing values handled. 
        """
        pass

# Strategy for Dropping the Missing Values
class DropMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, axis=0, thresh=None):
        """
        Initializes the DropMissingValuesStrategy with Specific parameters.

        Parameters:
        axis (int): 0 to drop rows with missing values, 1 to drop columns with missing values. 

        thresh (int): Threshold for non-NA values. Rows/Columns with less than threshold non-NA values are dropped. 
        """
        self.axis= axis
        self.thresh= thresh
    
    def handle(self, df: pd.DataFrame)-> pd.DataFrame:
        """
        Drops the Rows/Columns with missing values based on the axis and threshold. 

        Parameters: 
        df (pd.DataFrame): The input DataFrame which contains the missing values. 

        Returns:
        pd.DataFrame: The DataFrame with missing values dropped.
        """
        logging.info(f"Dropping the missing values with axis={self.axis} and thresh={self.thresh}")
        df_cleaned= df.dropna(axis=self.axis, thresh=self.thresh)
        logging.info("Missing values dropped")
        return df_cleaned


# Strategy for Filling the Missing Values
class FillingMissingValuesStrategy(MissingValueHandlingStrategy):
    def __init__(self, method="mean", fill_value=None):
        """
        Filling the Missing values using the specified method or constant value. 
        
        Parameters:
        df (df.DataFrame): The input DataFrame containing missing values. 

        Returns:
        pd.DataFrame:The DataFrame with missing values filled. 
        """
        self.method=method
        self.fill_value= fill_value

    def handle(self, df: pd.DataFrame)-> pd.DataFrame:
        """
        Filling missing values using the specific method or constant value. 

        Parameters:
        df (df.DataFrame): The input DataFrame containing missing values. 

        Returns:
        pd.DataFrame:The DataFrame with missing values filled. 
        """
        logging.info(f"Filling missing vlaues using method {self.method}")

        df_cleaned= df.copy()

        if self.method== "mean":
            numeric_columns= df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns]= df_cleaned[numeric_columns].fillna(df[numeric_columns].mean())

        elif self.method == "median":
            numeric_columns=df_cleaned.select_dtypes("number").columns
            df_cleaned[numeric_columns]=df_cleaned[numeric_columns].fillna(df[numeric_columns].median())
        
        elif self.method == "mode":
            for column in df_cleaned.columns:
                df_cleaned[column].fillna(df[column].mode().iloc[0],inplace=True)
        
        elif self.method== "constant":
            df_cleaned=df_cleaned.fillna(self.fill_value)
        
        else:
            logging.info(f"Unknown method: {self.method}. No missing value handled. ")

        logging.info("Missing values are handled. ")
        return df_cleaned



# Context Class for Handling the Missing Values
class MissingValueHandler:
    def __init__(self, strategy: MissingValueHandlingStrategy):
        """
        Initializes the MissingValueHandler with a Specific missing value handling strategy. 

        Parameters:
        Strategy (MissingValueHandlingStrategy): The Strategy to be used for handling missing values. 
        """
        self._strategy = strategy
    
    def set_strategy(self, strategy: MissingValueHandlingStrategy):
        """
        Set a new strategy for the MissingValueHandler. 

        Parameters:
        Strategy ( MissingvalueHandlingStategy): The new strategy to be used for handling missing values. 
        """
        logging.info("Choosing the missing value handling stategy")
        self._strategy= strategy
    
    def handling_missing_value(self, df:pd.DataFrame)-> pd.DataFrame:
        """
        Executes the missing value handling using the current Strategy.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values. 

        Return: 
        The DataFrame with missing values handled. 
        """
        logging.info("Executing missing value handling stategy")
        return self._strategy.handle(df)
    

# Example usage
if __name__ == "__main__":
    # Example dataframe
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Initialize missing value handler with a specific strategy
    # missing_value_handler = MissingValueHandler(DropMissingValuesStrategy(axis=0, thresh=3))
    # df_cleaned = missing_value_handler.handle_missing_values(df)

    # Switch to filling missing values with mean
    # missing_value_handler.set_strategy(FillMissingValuesStrategy(method='mean'))
    # df_filled = missing_value_handler.handle_missing_values(df)

    pass

        
        