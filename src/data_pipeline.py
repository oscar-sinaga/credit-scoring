from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import os
import copy
import util as util
import numpy as np

def read_raw_data(config: dict) -> pd.DataFrame:
    # Create variable to store raw dataset
    raw_dataset = pd.DataFrame()

    # Raw dataset dir
    raw_dataset_dir = config["raw_dataset_dir"]

    # Look and load add CSV files
    for i in tqdm(os.listdir(raw_dataset_dir)):
        raw_dataset = pd.concat([pd.read_csv(raw_dataset_dir + i), raw_dataset])
    
    # Return raw dataset
    return raw_dataset

def remove_outliers(df, col):
    """
    Menghapus outlier dari DataFrame berdasarkan metode IQR.
    
    Parameters:
        df (DataFrame): DataFrame yang akan dihapus outlier-nya.
    
    Returns:
        DataFrame: DataFrame baru yang telah dihapus outlier-nya.
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_no_outlier = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df_no_outlier

def check_data(input_data, params, api = False):
    input_data = copy.deepcopy(input_data)
    params = copy.deepcopy(params)

    if not api:
        # Check data types
        assert input_data.select_dtypes("object").columns.to_list() == \
            params["object_columns"], "an error occurs in object column(s)."
        assert input_data.select_dtypes("int").columns.to_list() == \
            params["int64_columns"], "an error occurs in int64_columns column(s)."
        assert input_data.select_dtypes("float").columns.to_list() == \
            params["float_columns"], "an error occurs in float_columns column(s)."
        # Check data_log range
        assert input_data.monthly_income_log.between(params["range_monthly_income_log"][0], params["range_monthly_income_log"][1]).sum() == \
        len(input_data), "an error occurs in monthly_income_log range."
        assert input_data.num_of_dependent_log.between(params["range_num_of_dependent_log"][0], params["range_num_of_dependent_log"][1]).sum() == \
            len(input_data), "an error occurs in num_of_dependent_log range."
        assert input_data.lama_bekerja_log.between(params["range_lama_bekerja_log"][0], params["range_lama_bekerja_log"][1]).sum() == \
            len(input_data), "an error occurs in lama_bekerja_log range."
    else:
        # In case checking data from api
        # Predictor that has object dtype only stasiun
        object_columns = params["object_columns"][:-1]

        # Max column not used as predictor
        int_columns = params["int64_columns"]

        # Check data types
        assert input_data.select_dtypes("object").columns.to_list() == \
            object_columns, "an error occurs in object column(s)."
        assert input_data.select_dtypes("int").columns.to_list() == \
            int_columns, "an error occurs in int32 column(s)."

    assert set(input_data.housing_type).issubset(set(params["range_housing_type"])), \
        "an error occurs in housing_type range."
    assert set(input_data.status_pernikahan).issubset(set(params["range_status_pernikahan"])), \
        "an error occurs in status_pernikahan range."
    assert set(input_data.pekerjaan).issubset(set(params["range_pekerjaan"])), \
        "an error occurs in pekerjaan range."
    assert set(input_data.housing_type).issubset(set(params["range_housing_type"])), \
        "an error occurs in housing_type range."
    
    assert input_data.monthly_income.between(params["range_monthly_income"][0], params["range_monthly_income"][1]).sum() == \
        len(input_data), "an error occurs in monthly_income range."
    assert input_data.num_of_dependent.between(params["range_num_of_dependent"][0], params["range_num_of_dependent"][1]).sum() == \
        len(input_data), "an error occurs in num_of_dependent range."
    assert input_data.lama_bekerja.between(params["range_lama_bekerja"][0], params["range_lama_bekerja"][1]).sum() == \
        len(input_data), "an error occurs in lama_bekerja range."
    assert input_data.otr.between(params["range_otr"][0], params["range_otr"][1]).sum() == \
        len(input_data), "an error occurs in otr range."
    assert input_data.tenor.between(params["range_tenor"][0], params["range_tenor"][1]).sum() == \
        len(input_data), "an error occurs in tenor range."
    assert input_data.dp.between(params["range_dp"][0], params["range_dp"][1]).sum() == \
        len(input_data), "an error occurs in dp range."
    
    
if __name__ == "__main__":
    # 1. Load configuration file
    config_data = util.load_config()

    # 2. Read all raw dataset
    raw_dataset = read_raw_data(config_data)

    # 3. Reset index
    raw_dataset.reset_index(
        inplace = True,
        drop = True
    )

    # 4. Save raw dataset
    util.pickle_dump(
        raw_dataset,
        config_data["raw_dataset_path"]
    )



    # 5. # Handling Variabel "paid_date" dan "due_date"
    raw_dataset['paid_date'] = pd.to_datetime(raw_dataset['paid_date'] )
    raw_dataset['due_date'] = pd.to_datetime(raw_dataset['due_date'] )

    # 6. # Handling Variabel `due_amount`, `paid_amount`, `monthly_income`, `otr`, `dp`, `num_of_dependent` dan `lama_bekerja` ke bentuk int
    raw_dataset['num_of_dependent'] = raw_dataset['num_of_dependent'].str.replace(' orang', '').str.replace('.','').astype('int64')
    raw_dataset['lama_bekerja'] = raw_dataset['lama_bekerja'].str.replace(' tahun', '').str.replace('.','').astype('int64')
    raw_dataset['due_amount'] = raw_dataset['due_amount'].str.replace('Rp', '').str.replace('.','').astype('int64')
    raw_dataset['paid_amount'] = raw_dataset['paid_amount'].str.replace('Rp', '').str.replace('.','').astype('int64')
    raw_dataset['monthly_income'] = raw_dataset['monthly_income'].str.replace('Rp', '').str.replace('.','').astype('int64')
    raw_dataset['otr'] = raw_dataset['otr'].str.replace('Rp', '').str.replace('.','').astype('int64')
    raw_dataset['dp'] = raw_dataset['dp'].str.replace('Rp', '').str.replace('.','').astype('int64')

    # 7. Handling data labelling
        #Pertama kita coba buat kolom selisih hari yaitu `due_date - paid_date `
    raw_dataset['selisih_hari'] = (raw_dataset['due_date'] - raw_dataset['paid_date']).dt.days

        # Membuat kolom baru dengan nilai 1 jika NPl dan 0 jika tidak
    raw_dataset['NPL'] =  'Tidak'
    raw_dataset.loc[raw_dataset['due_amount'] > 0, 'NPL'] = 'Ya'
    raw_dataset.loc[(raw_dataset['due_amount'] == 0) & (raw_dataset['selisih_hari'] <= -90), 'NPL'] = 'Ya'
        # Remove column ['app_id','due_date','paid_date','due_amount','paid_amount','selisih_hari']
    raw_dataset = raw_dataset.drop(columns=['app_id','due_date','paid_date','due_amount','paid_amount','selisih_hari'],axis=1)

    # 6. # Handling skewed data with log Transformation
    for col in config_data['columns_being_logtransformed']:
        col_log = f'{col}_log'
        raw_dataset[col_log] = raw_dataset[col].apply(lambda x: np.log(x+1))
            

    # 7. Handling Outlier
    list_df = []
    for col in raw_dataset.columns:
        if raw_dataset[col].dtype != 'O' and col != 'dp' and  col not in config_data['columns_being_logtransformed']:
            outlier_removed = remove_outliers(raw_dataset, col)
            list_df.append(outlier_removed)

    # Ambil indeks dari DataFrame pertama dalam list_df
    common_index = list_df[0].index

    # Iterasi melalui DataFrame lainnya dalam list_df untuk mendapatkan indeks yang bersamaan
    for df in list_df[1:]:
        common_index = common_index.intersection(df.index)

    raw_dataset = raw_dataset.loc[common_index]

    # 8. Check data definition
    check_data(raw_dataset, config_data)

    # 9. Splitting input output
    x = raw_dataset[config_data["predictors"]].copy()
    y = raw_dataset[config_data['label']].copy()

    # 14. Splitting train test
    x_train, x_test, \
    y_train, y_test = train_test_split(
        x, y,
        test_size = 0.3,
        random_state = 42,
        stratify = y
    )

    # 15. Splitting test valid
    x_valid, x_test, \
    y_valid, y_test = train_test_split(
        x_test, y_test,
        test_size = 0.5,
        random_state = 42,
        stratify = y_test
    )

    # 16. Save train, valid and test set
    util.pickle_dump(x_train, config_data["train_set_path"][0])
    util.pickle_dump(y_train, config_data["train_set_path"][1])

    util.pickle_dump(x_valid, config_data["valid_set_path"][0])
    util.pickle_dump(y_valid, config_data["valid_set_path"][1])

    util.pickle_dump(x_test, config_data["test_set_path"][0])
    util.pickle_dump(y_test, config_data["test_set_path"][1])