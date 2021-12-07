# This is the data exploration file for the kaggle competition

# In this file I will do feature engineering. Such as dummy
# variable creation for categorical variables, creating fewer
# categories for highly related categories, in terms of home type
# etc., for continuous variables we may scale the data.
# I will examine relationships withing the data using FAMD,
# correlation scatter plots, and histograms to find the distributions
# of the data.

import pandas as pd
import matplotlib.pyplot as plt
import prince
import category_encoders as ce
import numpy as np
from sklearn.preprocessing import StandardScaler


class Analysis:

    # Initialize the class to hold the data frame
    def __init__(self, data: pd.DataFrame) -> None:
        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            raise TypeError()

    # Returns the dataframe
    def get_data(self) -> pd.DataFrame:
        return self.data


# I will use this class to hold the data and perform exploratory analysis
class ExploratoryAnalysis(Analysis):

    # Perform FAMD, plot and save the chart
    def dim_reduce(self, url: str, decision: str, dim_reduce=prince.FAMD) -> None:
        # Run the dimension reduction algo
        dim_reducer = dim_reduce(n_components=4, n_iter=5, random_state=0)
        dim_reducer.fit(self.data.drop(decision, axis=1))
        # Plot the dimension reduction
        dim_reducer.plot_row_coordinates(
            self.data.drop(decision, axis=1),
            figsize=(15, 10),
            color_labels=['Rented' if dec == 1 else 'Vacant' for dec in self.data[decision]]
        )
        # Save the figure
        plt.savefig(url)
        plt.show()

    # create the correlation scatter plot matrix.
    # Contains histograms on the diagonal
    def corr_scatter(self, url: str) -> None:
        pd.plotting.scatter_matrix(self.data, figsize=(20, 12))
        plt.savefig(url)
        plt.show()


# This class encodes the categorical variables scales the continuous variables
# and recodes categorical variables into fewer categories.
class FeatureEngineering(Analysis):

    # Encode categorical variables
    def categorical_variables(self, cols: list, encoder=ce.HashingEncoder):

        if isinstance(encoder, ce.HashingEncoder):
            num = int(len(self.data[cols].drop_duplicates().index) * .8)
            ce_encoder = encoder(cols=cols, n_components=num)
        else:
            ce_encoder = encoder(cols=cols)
        x = ce_encoder.fit_transform(self.data[cols])
        self.data = pd.concat([self.data.drop(cols, axis=1), x], axis=1)
        return ce_encoder

    # Scale continuous variables
    def scale(self, cols: list, transform=StandardScaler) -> None:
        self.data[cols] = transform().fit_transform(self.data[cols])

    # Map strings to categories
    def regroup_features(self, cols: list, group_dict: dict) -> None:
        self.data[list(cols)] = np.array([self.data[col].map(group_dict[col]) for col in group_dict.keys()]).T

    # Remove non numeric characters from a string
    def remove_nonnumeric(self, col):
        self.data[col] = [
            float(
                ''.join([x for x in self.data.loc[idx, col] if x.isnumeric()])
            ) for idx in self.data.index
        ]
