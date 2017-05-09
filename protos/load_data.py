import os
import pickle
import pandas
import numpy
from IPython.core.display import display
TRAIN_DATA_FILE = '../data/train.csv'
TEST_DATA_FILE = '../data/test.csv'
MACRO_DATA_FILE = '../data/macro.csv'
from sklearn import model_selection, preprocessing


def load_macro_data():
    df = pandas.read_csv(MACRO_DATA_FILE, parse_dates=['timestamp'])
    #df['timestamp'] = df['timestamp'].apply(lambda x: int(x.replace('-', '')))

    cols = [col for col in df.columns.values
            if col not in df.describe().columns.values]
    for col in cols:
        df[col] = df[col].apply(lambda x: float(x.replace(',', '')) if ',' in str(x) else None)
    df_macro = df  # .sort_values('timestamp').fillna(method='ffill').fillna(method='bfill')

    return df_macro


def load_train_data2():
    df = pandas.read_csv(TRAIN_DATA_FILE, parse_dates=['timestamp'])

    #df_macro = load_macro_data()
    #df = pandas.merge_ordered(df, df_macro, on='timestamp', how='left')

    labels = df['price_doc'].values
    df = df.drop(["id", "timestamp", "price_doc"], axis=1)
    for c in df.columns:
        if df[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(list(df[c].values))
    df = df.fillna(-100)
    return df, labels


def load_test_data2(cols):
    df = pandas.read_csv(TEST_DATA_FILE, parse_dates=['timestamp'])

    #df_macro = load_macro_data()
    #df = pandas.merge_ordered(df, df_macro, on='timestamp', how='left')

    df = df.drop(["id", "timestamp"], axis=1)
    for c in df.columns:
        if df[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(list(df[c].values))
    for col in cols:
        if col not in df.columns.values:
            print('MISSING!', col)
            df[col] = numpy.zeros(df.shape[0])
    df = df.fillna(-100)
    return df[cols]


def load_train_data():
    #df_macro = load_macro_data()
    df = pandas.read_csv(TRAIN_DATA_FILE, parse_dates=['timestamp'])
    labels = df['price_doc'].values

    df.drop(['id', 'price_doc'], axis=1, inplace=True)
    """
    month_year = (df.timestamp.dt.month + df.timestamp.dt.year * 100)
    month_year_cnt_map = month_year.value_counts().to_dict()
    df['month_year_cnt'] = month_year.map(month_year_cnt_map)

    week_year = (df.timestamp.dt.weekofyear + df.timestamp.dt.year * 100)
    week_year_cnt_map = week_year.value_counts().to_dict()
    df['week_year_cnt'] = week_year.map(week_year_cnt_map)

    week_day = df.timestamp.dt.weekday
    tmp = week_day.value_counts()
    week_day_cnt_map = (tmp / tmp.sum()).to_dict()
    df['weekday_cnt_rate'] = week_day.map(week_day_cnt_map)

    df['month'] = df.timestamp.dt.month
    #df['weekday'] = df.timestamp.dt.weekday
    df['dow'] = df.timestamp.dt.dayofweek
    """
    # Other feature engineering
    df['rel_floor'] = df['floor'] / df['max_floor'].astype(float)
    df['rel_kitch_sq'] = df['kitch_sq'] / df['full_sq'].astype(float)
    df.drop(['timestamp'], axis=1, inplace=True)

    df['product_type'] = df['product_type'].apply(lambda x: x == 'Investment')
    #area = pandas.get_dummies(df['sub_area'], prefix='sub_area')
    # for col in area:
    #    df[col] = area[col]
    for col in ['culture_objects_top_25', 'thermal_power_plant_raion', 'incineration_raion', 'oil_chemistry_raion',
                'radiation_raion', 'railroad_terminal_raion', 'big_market_raion', 'nuclear_reactor_raion',
                'detention_facility_raion', 'water_1line', 'big_road1_1line', 'railroad_1line']:
        df[col] = df[col].apply(lambda x: x == 'yes')
    map_ecology = {a: i for i, a in enumerate(['no data', 'poor', 'satisfactory', 'good', 'excellent'])}
    df['ecology'] = df['ecology'].apply(map_ecology.get)
    df.drop(['sub_area'], axis=1, inplace=True)
    df = df.fillna(-100)
    return df, labels
    """
    df_numeric = df.select_dtypes(exclude=['object'])
    df_obj = df.select_dtypes(include=['object']).copy()
    for c in df_obj:
        df_obj[c] = pandas.factorize(df_obj[c])[0]

    df_values = pandas.concat([df_numeric, df_obj], axis=1)

    return df_values, labels
    """


def load_test_data(cols):
    #df_macro = load_macro_data()
    df = pandas.read_csv(TEST_DATA_FILE, parse_dates=['timestamp'])
    #df = pandas.merge_ordered(df, df_macro, on='timestamp', how='left')

    df.drop(['id'], axis=1, inplace=True)
    """
    month_year = (df.timestamp.dt.month + df.timestamp.dt.year * 100)
    month_year_cnt_map = month_year.value_counts().to_dict()
    df['month_year_cnt'] = month_year.map(month_year_cnt_map)

    week_year = (df.timestamp.dt.weekofyear + df.timestamp.dt.year * 100)
    week_year_cnt_map = week_year.value_counts().to_dict()
    df['week_year_cnt'] = week_year.map(week_year_cnt_map)

    week_day = df.timestamp.dt.weekday
    tmp = week_day.value_counts()
    week_day_cnt_map = (tmp / tmp.sum()).to_dict()
    df['weekday_cnt_rate'] = week_day.map(week_day_cnt_map)

    df['month'] = df.timestamp.dt.month
    #df['weekday'] = df.timestamp.dt.weekday
    df['dow'] = df.timestamp.dt.dayofweek
    """
    # Other feature engineering
    df['rel_floor'] = df['floor'] / df['max_floor'].astype(float)
    df['rel_kitch_sq'] = df['kitch_sq'] / df['full_sq'].astype(float)
    df.drop(['timestamp'], axis=1, inplace=True)

    df['product_type'] = df['product_type'].apply(lambda x: x == 'Investment')
    #area = pandas.get_dummies(df['sub_area'], prefix='sub_area')
    # for col in area:
    #    df[col] = area[col]
    for col in ['culture_objects_top_25', 'thermal_power_plant_raion', 'incineration_raion', 'oil_chemistry_raion',
                'radiation_raion', 'railroad_terminal_raion', 'big_market_raion', 'nuclear_reactor_raion',
                'detention_facility_raion', 'water_1line', 'big_road1_1line', 'railroad_1line']:
        df[col] = df[col].apply(lambda x: x == 'yes')
    map_ecology = {a: i for i, a in enumerate(['no data', 'poor', 'satisfactory', 'good', 'excellent'])}
    df['ecology'] = df['ecology'].apply(map_ecology.get)
    df.drop(['sub_area'], axis=1, inplace=True)
    df = df.fillna(-100)
    for col in cols:
        if col not in df.columns.values:
            print('MISSING!', col)
            df[col] = numpy.zeros(df.shape[0])

    return df[cols]
    """
    df_numeric = df.select_dtypes(exclude=['object'])
    df_obj = df.select_dtypes(include=['object']).copy()
    for c in df_obj:
        df_obj[c] = pandas.factorize(df_obj[c])[0]

    df_values = pandas.concat([df_numeric, df_obj], axis=1)
    return df_values
    """

if __name__ == '__main__':
    load_train_data()
