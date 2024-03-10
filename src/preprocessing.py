import pandas as pd
import numpy as np
import util as util
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

def load_dataset(config_data: dict) -> pd.DataFrame:
    # Load every set of data
    x_train = util.pickle_load(config_data["train_set_path"][0])
    y_train = util.pickle_load(config_data["train_set_path"][1])

    x_valid = util.pickle_load(config_data["valid_set_path"][0])
    y_valid = util.pickle_load(config_data["valid_set_path"][1])

    x_test = util.pickle_load(config_data["test_set_path"][0])
    y_test = util.pickle_load(config_data["test_set_path"][1])

    # Concatenate x and y each set
    train_set = pd.concat(
        [x_train, y_train],
        axis = 1
    )
    valid_set = pd.concat(
        [x_valid, y_valid],
        axis = 1
    )
    test_set = pd.concat(
        [x_test, y_test],
        axis = 1
    )

    # Return 3 set of data
    return train_set, valid_set, test_set

def join_label_categori(set_data, config_data):
    # Check if label not found in set data
    if config_data["label"] in set_data.columns.to_list():
        # Create copy of set data
        set_data = set_data.copy()
        return set_data
    else:
        raise RuntimeError("Kolom label tidak terdeteksi pada set data yang diberikan!")

def nan_detector(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Replace -1 with NaN
    set_data.replace(
        -1, np.nan,
        inplace = True
    )

    # Return replaced set data
    return set_data

def ohe_fit(data_tobe_fitted: dict, ohe_path: str) -> OneHotEncoder:
    # Create ohe object
    ohe = OneHotEncoder(sparse_output = False)

    # Fit ohe
    ohe.fit(np.array(data_tobe_fitted).reshape(-1, 1))

    # Save ohe object
    util.pickle_dump(ohe, f'{ohe_path}.pkl')

    # Return trained ohe
    return ohe

def ohe_fit_all(ohe_path: str) -> OneHotEncoder:
    ohe_fit(config_data["range_housing_type"], f"{ohe_path}/ohe_housing_type")
    ohe_fit(config_data["range_status_pernikahan"], f"{ohe_path}/ohe_status_pernikahan")
    ohe_fit(config_data["range_pekerjaan"], f"{ohe_path}/ohe_pekerjaan")


def ohe_transform(set_data: pd.DataFrame, tranformed_column: str, ohe_path: str) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    ohe_path = f"{ohe_path}/ohe_{tranformed_column}.pkl"

    # Load ohe categorical
    ohe_statiun = util.pickle_load(ohe_path)

    # Transform variable categorical of set data, resulting array
    categorical_features = ohe_statiun.transform(np.array(set_data[tranformed_column].to_list()).reshape(-1, 1))

    # Convert to dataframe
    categorical_features = pd.DataFrame(categorical_features.tolist(), columns = list(ohe_statiun.categories_[0]))

    # Set index by original set data index
    categorical_features.set_index(set_data.index, inplace = True)

    # Concatenate new features with original set data
    set_data = pd.concat([categorical_features, set_data], axis = 1)

    # Drop categorical column
    set_data.drop(columns = tranformed_column, inplace = True)

    # Convert columns type to string
    new_col = [str(col_name) for col_name in set_data.columns.to_list()]
    set_data.columns = new_col

    # Return new feature engineered set data
    return set_data

def ohe_transform_all(set_data,config_data):
    result = ohe_transform(set_data, "housing_type", config_data["ohe_path"])
    result = ohe_transform(result, "status_pernikahan", config_data["ohe_path"])
    result = ohe_transform(result, "pekerjaan", config_data["ohe_path"])
    return result


def rus_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Create sampling object
    rus = RandomUnderSampler(random_state = 26)

    # Balancing set data
    x_rus, y_rus = rus.fit_resample(
        set_data.drop("NPL", axis = 1),
        set_data.NPL
    )

    # Concatenate balanced data
    set_data_rus = pd.concat(
        [x_rus, y_rus],
        axis = 1
    )

    # Return balanced data
    return set_data_rus

def ros_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Create sampling object
    ros = RandomOverSampler(random_state = 11)

    # Balancing set data
    x_ros, y_ros = ros.fit_resample(
        set_data.drop("NPL", axis = 1),
        set_data.NPL
    )

    # Concatenate balanced data
    set_data_ros = pd.concat(
        [x_ros, y_ros],
        axis = 1
    )

    # Return balanced data
    return set_data_ros

def sm_fit_resample(set_data: pd.DataFrame) -> pd.DataFrame:
    # Create copy of set data
    set_data = set_data.copy()

    # Create sampling object
    sm = SMOTE(random_state = 112)

    # Balancing set data
    x_sm, y_sm = sm.fit_resample(
        set_data.drop("NPL", axis = 1),
        set_data.NPL
    )

    # Concatenate balanced data
    set_data_sm = pd.concat(
        [x_sm, y_sm],
        axis = 1
    )

    # Return balanced data
    return set_data_sm

def le_fit(data_tobe_fitted: dict, le_path: str) -> LabelEncoder:
    # Create le object
    le_encoder = LabelEncoder()

    # Fit le
    le_encoder.fit(data_tobe_fitted)

    # Save le object
    util.pickle_dump(
        le_encoder,
        le_path
    )

    # Return trained le
    return le_encoder

def le_transform(label_data: pd.Series, config_data: dict) -> pd.Series:
    # Create copy of label_data
    label_data = label_data.copy()

    # Load le encoder
    le_encoder = util.pickle_load(config_data["le_encoder_path"])

    # If categories both label data and trained le matched
    if len(set(label_data.unique()) - set(le_encoder.classes_) | set(le_encoder.classes_) - set(label_data.unique())) == 0:
        # Transform label data
        label_data = le_encoder.transform(label_data)
    else:
        raise RuntimeError("Check category in label data and label encoder.")
    
    # Return transformed label data
    return label_data


def delete_columns_being_logtransformed(set_data,config_data):
    result = set_data.drop(config_data['columns_being_logtransformed'],axis=1)
    return result

def log_transform(set_data,config_data):
    columns_to_transform = config_data['columns_being_logtransformed']
    for col in columns_to_transform:
       col_log = f'{col}_log'
       set_data[col_log] = set_data[col].apply(lambda x: np.log(x+1))
    set_data = delete_columns_being_logtransformed(set_data,config_data)
    return set_data

if __name__ == "__main__":
    # 1. Load configuration file
    config_data = util.load_config()

    # 2. Load dataset
    train_set, valid_set, test_set = load_dataset(config_data)

    # 3. Join label categories
    train_set = join_label_categori(
        train_set,
        config_data
    )
    valid_set = join_label_categori(
        valid_set,
        config_data
    )
    test_set = join_label_categori(
        test_set,
        config_data
    )

    # 4. Delete columns_being_logtransformed
    train_set = delete_columns_being_logtransformed(train_set,config_data)
    valid_set = delete_columns_being_logtransformed(valid_set,config_data)
    test_set = delete_columns_being_logtransformed(test_set,config_data)

    # 5. Converting -1 to NaN
    train_set = nan_detector(train_set)
    valid_set = nan_detector(valid_set)
    test_set = nan_detector(test_set)

    # Fit ohe with categorical data
    ohe_fit_all(config_data['ohe_path'])
    
    # 9. Transform NPL on train, valid, and test set
    train_set = ohe_transform_all(train_set,config_data)

    valid_set = ohe_transform_all(valid_set,config_data)

    test_set = ohe_transform_all(test_set,config_data)

    # 10. Undersampling dataset
    train_set_rus = rus_fit_resample(train_set)

    # 11. Oversampling dataset
    train_set_ros = ros_fit_resample(train_set)

    # 12. SMOTE dataset
    train_set_sm = sm_fit_resample(train_set)

    # 13. Fit label encoder
    le_encoder = le_fit(
        config_data["label_categories"],
        config_data["le_encoder_path"]
    )

    # 14. Label encoding undersampling set
    train_set_rus.NPL = le_transform(
        train_set_rus.NPL, 
        config_data
    )

    # 15. Label encoding overrsampling set
    train_set_ros.NPL = le_transform(
        train_set_ros.NPL,
        config_data
    )

    # 16. Label encoding smote set
    train_set_sm.NPL = le_transform(
        train_set_sm.NPL,
        config_data
    )

    # 17. Label encoding validation set
    valid_set.NPL = le_transform(
        valid_set.NPL,
        config_data
    )

    # 18. Label encoding test set
    test_set.NPL = le_transform(
        test_set.NPL,
        config_data
    )

    # 19. Dumping dataset
    x_train = {
        "Undersampling" : train_set_rus.drop(columns = "NPL"),
        "Oversampling" : train_set_ros.drop(columns = "NPL"),
        "SMOTE" : train_set_sm.drop(columns = "NPL")
    }

    y_train = {
        "Undersampling" : train_set_rus.NPL,
        "Oversampling" : train_set_ros.NPL,
        "SMOTE" : train_set_sm.NPL
    }

    util.pickle_dump(
        x_train,
        "data/processed/x_train_feng.pkl"
    )
    util.pickle_dump(
        y_train,
        "data/processed/y_train_feng.pkl"
    )

    util.pickle_dump(
        valid_set.drop(columns = "NPL"),
        "data/processed/x_valid_feng.pkl"
    )
    util.pickle_dump(
        valid_set.NPL,
        "data/processed/y_valid_feng.pkl"
    )

    util.pickle_dump(
        test_set.drop(columns = "NPL"),
        "data/processed/x_test_feng.pkl"
    )
    util.pickle_dump(
        test_set.NPL,
        "data/processed/y_test_feng.pkl"
    )