import numpy as np
import pandas as pd
from logger import logging 
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


## Abstract Base Class for Feature Engineering Strategy
# ---------------------------------------------------
# This class defines a common interface for different feature engineering strategies. 
# Subclasses must implement the apply_transformation method. 

class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df:pd.DataFrame)->pd.DataFrame:
        """
        Abstract method to apply the Feature Engineering transformation to the DataFrame.

        Parameters:
        df (df.DataFrame): The DataFrame containing features to transform. 

        Returns:
        pd.DataFrame: A DataFrame with the applied transformation. 
        """
        pass


## Concrete Strategy for Log Transformation
# ---------------------------------------------------
# This strategy applies a logarithmic transformation to skewed features to normalize the distribution. 

class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the LogTransformation with the specific features to transform. 

        Parameters: 
        features (list): The list of features to apply the log transformation to
        """
        self.features = features

    def apply_transformation(self, df:pd.DataFrame)->pd.DataFrame:
        """
        Applies a log transformation to specified features in the DataFrame. 

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The DataFrame with log-tranformed features. 
        """
        logging.info(f"Applying log transformation to features:{self.features} ")
        df_transformed= df.copy()
        for feature in self.features:
            df_transformed[feature]=np.log1p(df[feature]) # log1p handles log(0) by calculating log(1+x)
        logging.info("Log Transformation completed")
        return df_transformed
    

# Concrete Strategy for Standard Scaling 
#----------------------------------------------------
# This Strategy applies Standard Scaling ( z-score normalization) to features, centring them around zero with unit variance. 

class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the StandardScaling with the specific features to scale. 

        Parameters:
        features (list): The list of features to apply the standard scaling to. 
        """        
        self.features=features
        self.scaler = StandardScaler()
    
    def apply_transformation(self, df: pd.DataFrame)->pd.DataFrame:
        """
        Applies standard scaling to the specified features in the DataFrame. 

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform. 

        Result:
        pd.DataFrame: The tranformed dataframe with applied Standard Scaling transformation. 
        """
        logging.info(f"Appling the Standard Scaling on the features: {self.features}")
        df_tranformed = df.copy()
        df_tranformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Standard Scaling is completed on the features.")
        return df_tranformed


## Concrete class for Min-Max Scaling
#---------------------------------------------------
# This strategy applies the Min-max scalling to the features, scaling them to a specific range, typically [0,1]. 

class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features, feature_range=(0,1)):
        """
        Initializes the MinMaxScaling with the specific features to scale and target range. 

        Parameters:
        features (list): List of features to apply the Min-max scaling. 

        feature_range (tuple): Target range for scaling, default is (0,1). 
        """
        self.features=features
        self.scaler=MinMaxScaler(feature_range=feature_range)
    
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Min-Max scaling to the provided features on the DataFrame. 

        Parameters:
        df (pd.DataFrame): The Dataframe containing features to transform. 

        Returns:
        pd.DataFrame: The dataframe with Min-max scaled output. 
        """
        logging.info("Applying the Min-max scaling on the given features.")
        df_tranformed = df.copy()
        df_tranformed[self.features] = self.scaler.fit_transform(df[self.features])
        logging.info("Min-max scaling completed on the features")
        return df_tranformed
    

## Concrete Stategy for One-hot encoding
# ---------------------------------------------------
# This strategy applies one-hot encoding to categorical features, converting them into binary vectors. 

class OneHotEncodingf(FeatureEngineeringStrategy):
    def __init__(self, features):

        """
        Initializes the OneHotEncoding with the specific features to encode. 

        Parameters:
        features (list): List of categorical features to apply the one-hot encodinng. 
        """
        self.features = features
        self.encoder= OneHotEncoder(sparse=False, drop="first")
    
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies one-hot encoding to the specified categorical features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with one-hot encoded features.
        """
        logging.info(f"Starting One-hot encoding on the provided features")
        df_transformed=df.copy()
        encoded_df=pd.DataFrame(self.encoder.fit_transform(df[self.features]),columns=self.encoder.get_feature_names_out(self.features),)
        df_transformed[self.features] = df_transformed.drop(columns=self.features).reset_index(drop=True)
        df_transformed=pd.concat([df_transformed,encoded_df],axis=1)
        logging.info("One-hot encoding completed")
        return df_transformed


# Context class for Feature Engineering
## --------------------------------------------------
# This class uses a featureEngineeringStrategy to apply transformations to dataset. 

class FeatureEngineer:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        """
        Initializes the FeatureEngineer with a specific feature engineering stategy. 

        Parameter: 
        strategy (FeatureEngineeringStrategy): The Strategy to be used for feature enginneering.
        """
        self.strategy=strategy
    
    def set_strategy(self,strategy: FeatureEngineeringStrategy):
        """
        Sets a new strategy for the FeatureEngineer.

        Parameters:
        strategy (FeatureEngineeringStrategy): The new strategy to be used for feature engineering.
        """
        logging.info("Setting up the feature strategy")
        self._strategy = strategy

    def apply_feature_engineering(self,df:pd.DataFrame)->pd.DataFrame:
        """
        Applies the choosen feature engineering strategy on the DataFrame. 

        Parameter:
        pd.DataFrame: DataFrame with applied feature engineering transformation. 

        Result:
        pd.DataFrame: The dataframe with applied feature engineering transformations.
        """
        logging.info("Applying feature Engineering strategy.")
        return self._strategy.apply_transformation(df)
    



# Example usage
if __name__ == "__main__":
    # Example dataframe
    # df = pd.read_csv('../extracted-data/your_data_file.csv')

    # Log Transformation Example
    # log_transformer = FeatureEngineer(LogTransformation(features=['SalePrice', 'Gr Liv Area']))
    # df_log_transformed = log_transformer.apply_feature_engineering(df)

    # Standard Scaling Example
    # standard_scaler = FeatureEngineer(StandardScaling(features=['SalePrice', 'Gr Liv Area']))
    # df_standard_scaled = standard_scaler.apply_feature_engineering(df)

    # Min-Max Scaling Example
    # minmax_scaler = FeatureEngineer(MinMaxScaling(features=['SalePrice', 'Gr Liv Area'], feature_range=(0, 1)))
    # df_minmax_scaled = minmax_scaler.apply_feature_engineering(df)

    # One-Hot Encoding Example
    # onehot_encoder = FeatureEngineer(OneHotEncoding(features=['Neighborhood']))
    # df_onehot_encoded = onehot_encoder.apply_feature_engineering(df)

    pass  
