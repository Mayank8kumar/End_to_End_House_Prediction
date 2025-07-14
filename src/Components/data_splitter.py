import pandas as pd
from logger import logging
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split


## Abstract Base class for Data Splitting Strategy. 
#----------------------------------------------------------------
# This class defines a common interface for different data splitting strategies. 
# Subclasses must implement the split_data method. 

class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column:str):
        """
        Abstract method to split the data into training and testing sets.

        Parameters:
        df (pd.DataFrame): The input DataFrame to be split.
        target_column (str): The name of the target column.

        Returns:
        X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """
        pass


## Concrete Strategy for Simple Train-Test split
# -------------------------------------------------
# This strategy implemeents a simple train-test split. 
class SimpleTrainTestStrategy(DataSplittingStrategy):
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initializes the SimpleTrainTestSplitStrategy with specific parameters.

        Parameters:
        test_size (float): The proportion of the dataset to include in the test split. 

        Random_state (int): Use by the random nummber generator. 
        """
        self.test_size= test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_column: str):
        """
        Splits the data into training and testing set using a simple train-test split. 

        Parameters:
        df (DataFrame): the input DataFrame to be split. 
        target_column (str): The name of the target column. 

        Returns:
        X_train, X_test, y_train, y_test: The training and testing splits for features and target. 
        """
        logging.info("Performing simple train-test split.")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X,y, test_size=self.test_size, random_state=self.random_state
        )

        logging.info("Train-test split completed. ")
        return X_train, X_test, y_train, y_test
    
class DataSplitter:
    def __init__(self, strategy: DataSplittingStrategy):
        """
        Initializes the DataSplitter with a specific data splitting strategy. 

        Parameter:
        strategy (DataSplittingStrategy): The strategy to be used for data splitting.
        """
        self._strategy = strategy
    
    def set_strategy(self, strategy: DataSplittingStrategy):
        """
        Sets a new strategy for the DataSplitter.

        Parameters:
        strategy (DataSplittingStrategy): The new strategy to be used for data splitting.
        """
        logging.info("Swtiching data splitting strategy")
        self._strategy = strategy

    def split(self, df: pd.DataFrame, target_column: str):
        """
        Executes the data splitting using the current strategy.

        Parameters:
        df (pd.DataFrame): The input DataFrame to be split.
        target_column (str): The name of the target column.

        Returns:
        X_train, X_test, y_train, y_test: The training and testing splits for features and target.
        """
        logging.info("Splitting the data using selected strategy")
        return self._strategy.split_data(df, target_column)
    

# Example usage
if __name__ == "__main__":
    # Example dataframe (replace with actual data loading)
    # df = pd.read_csv('your_data.csv')

    # Initialize data splitter with a specific strategy
    # data_splitter = DataSplitter(SimpleTrainTestSplitStrategy(test_size=0.2, random_state=42))
    # X_train, X_test, y_train, y_test = data_splitter.split(df, target_column='SalePrice')

    pass
