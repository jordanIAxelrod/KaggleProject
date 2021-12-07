"""
This is the main file of the project. All of the function calls will come from here
The main things done in this project are exploratory analysis, model selection, training,
and prediction. The class definitions for these operations will be defined in separate files.
"""
import prince

import exploratory_analysis
import model_selection
import pandas as pd
from pandas.api.types import is_object_dtype, is_numeric_dtype
import category_encoders as ce
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


# Run the exploratory analysis feature engineering and model selection
def main():
    # Read in the data
    airbnb = pd.read_csv(r'Airbnb-raw/train.csv', index_col='id')

    # use zip codes as strings
    airbnb['Neighbourhood'] = airbnb['Neighbourhood'].astype(str)

    xpl = exploratory_analysis.ExploratoryAnalysis(airbnb.dropna())
    xpl.dim_reduce('Charts/FAMD.png', 'Decision')

    # True or False dictionary
    tf_dict = {
        't': 1,
        'f': 0
    }
    # dictionary for mapping qualitative features to quantitative
    mapping_dict = {
        "Bathrooms_text": {
            "1 bath": 1,
            "1 private bath": 1,
            "2 baths": 2,
            "1.5 baths": 1.5,
            "3 baths": 3,
            "2.5 baths": 2.5,
            "6 baths": 6,
            "5 baths": 5,
            "1 shared bath": 1,
            "3.5 baths": 3.5,
            "1.5 shared baths": 1.5,
            "4.5 baths": 4.5,
            "2.5 shared baths": 2.5,
            "4 baths": 4,
            "Private half-bath": 0.5,
            "Half-bath": 0.5,
            "9 baths": 9,
            "2 shared baths": 2,
            "6.5 baths": 6.5,
            "0 shared baths": 0,
            "0 baths": 0,
            "5.5 baths": 5.5,
            "7 baths": 7,
            '7.5 baths': 7.5
        },
        'Host_is_superhost': tf_dict,
        'Host_has_profile_pic': tf_dict,
        'Host_identity_verified': tf_dict,
        'Instant_bookable': tf_dict,
        'Host_response_time': {
            "within an hour": 1,
            "within a few hours": 2,
            "": 2.5,
            "within a day": 3,
            "a few days or more": 4

        }
    }
    test_data = pd.read_csv('Airbnb-raw/test.csv', index_col='id')

    # Create the feature engineering object
    feat_eng = exploratory_analysis.FeatureEngineering(test_data)
    # Make the price column into a numeric column
    feat_eng.remove_nonnumeric('Price')

    # Map the string features to numeric
    feat_eng.regroup_features(mapping_dict.keys(), mapping_dict)
    # return the data
    feat_eng.scale(
        [col for col in test_data.columns if is_numeric_dtype(test_data[col]) and col != 'Decision']
    )


    # Create the feature engineering object
    feat_eng = exploratory_analysis.FeatureEngineering(airbnb)
    # Make the price column into a numeric column
    feat_eng.remove_nonnumeric('Price')

    # Map the string features to numeric
    feat_eng.regroup_features(mapping_dict.keys(), mapping_dict)
    # return the data
    airbnb = feat_eng.get_data()
    feat_eng.scale(
        [col for col in airbnb.columns if is_numeric_dtype(airbnb[col]) and col != 'Decision']
    )
    cols = [col for col in airbnb.columns if is_object_dtype(airbnb[col])]
    encoder = feat_eng.categorical_variables(
        cols,
        ce.OneHotEncoder
    )
    airbnb = feat_eng.get_data()
    airbnb.loc[pd.isna(airbnb['Host_response_time']), 'Host_response_time'] = airbnb[
        'Host_response_time'].mean()
    airbnb.loc[pd.isna(airbnb['Host_is_superhost']), 'Host_is_superhost'] = 0
    airbnb.loc[pd.isna(airbnb['Host_has_profile_pic']), 'Host_has_profile_pic'] = 0
    airbnb.loc[pd.isna(airbnb['Host_identity_verified']), 'Host_identity_verified'] = 0
    airbnb.loc[pd.isna(airbnb['Bedrooms']), 'Bedrooms'] = airbnb.loc[
                                                              pd.isna(airbnb['Bedrooms']), 'Accommodates'] / 2
    airbnb.loc[pd.isna(airbnb['Beds']), 'Beds'] = airbnb.loc[pd.isna(airbnb['Beds']), 'Accommodates'] / 2
    airbnb.loc[pd.isna(airbnb['Review_scores_rating']), 'Review_scores_rating'] = airbnb[
        'Review_scores_rating'].mean()
    test_data['Neighbourhood'] = test_data['Neighbourhood'].astype(str)
    x = encoder.transform(test_data[cols])
    test_data = pd.concat([test_data.drop(cols, axis=1), x], axis=1)
    test_data.loc[pd.isna(test_data['Host_response_time']), 'Host_response_time'] = test_data[
        'Host_response_time'].mean()
    test_data.loc[pd.isna(test_data['Host_is_superhost']), 'Host_is_superhost'] = 0
    test_data.loc[pd.isna(test_data['Host_has_profile_pic']), 'Host_has_profile_pic'] = 0
    test_data.loc[pd.isna(test_data['Host_identity_verified']), 'Host_identity_verified'] = 0
    test_data.loc[pd.isna(test_data['Bedrooms']), 'Bedrooms'] = test_data.loc[
                                                                    pd.isna(test_data['Bedrooms']), 'Accommodates'] / 2
    test_data.loc[pd.isna(test_data['Beds']), 'Beds'] = test_data.loc[pd.isna(test_data['Beds']), 'Accommodates'] / 2
    test_data.loc[pd.isna(test_data['Review_scores_rating']), 'Review_scores_rating'] = test_data[
        'Review_scores_rating'].mean()


    print(feat_eng.get_data())
    xpl = exploratory_analysis.ExploratoryAnalysis(airbnb)
    xpl.dim_reduce('Charts/PCA.png', 'Decision', prince.PCA)
    # xpl.corr_scatter('Charts/Correlation_Scatter.png')
    print(xpl.get_data())
    model_selector = model_selection.Learner(xpl.get_data())
    best_params = model_selector.cross_validate('Decision')
    print(best_params)
    best_model = max(best_params, key=lambda x: x[1])

    mod = eval(best_model)()
    mod.set_params(**best_params[best_model][0])
    data = xpl.get_data()
    print(data.drop('Decision', axis='columns'))
    print(data['Decision'])
    mod.fit(data.drop('Decision', axis='columns'), data['Decision'])

    decision = pd.DataFrame(
        {
            'id': test_data.index,
            'Decision': mod.predict(test_data)
        }
    )
    print(decision)
    decision.to_csv('Output/Submission.csv', index=False)


if __name__ == '__main__':
    main()
