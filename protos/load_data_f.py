import os
import pickle
import pandas
import numpy
from IPython.core.display import display
TRAIN_DATA_FILE = '../data/train.csv'
TEST_DATA_FILE = '../data/test.csv'
MACRO_DATA_FILE = '../data/macro.csv'


def _load_train_data():
    df = pandas.read_csv(TRAIN_DATA_FILE)
    df['timestamp'] = df['timestamp'].apply(lambda x: int(x.replace('-', '')))
    df['product_type'] = df['product_type'].apply(lambda x: x == 'Investment')
    area = pandas.get_dummies(df['sub_area'], prefix='sub_area')
    for col in area:
        df[col] = area[col]
    for col in ['culture_objects_top_25', 'thermal_power_plant_raion', 'incineration_raion', 'oil_chemistry_raion',
                'radiation_raion', 'railroad_terminal_raion', 'big_market_raion', 'nuclear_reactor_raion',
                'detention_facility_raion', 'water_1line', 'big_road1_1line', 'railroad_1line']:
        df[col] = df[col].apply(lambda x: x == 'yes')
    map_ecology = {a: i for i, a in enumerate(['no data', 'poor', 'satisfactory', 'good', 'excellent'])}
    df['ecology'] = df['ecology'].apply(map_ecology.get)
    df_data = df.sort_values(['timestamp', 'sub_area', 'product_type']
                             ).fillna(method='ffill').fillna(method='bfill')
    FEATURE = [col for col in df.columns.values if col not in ['sub_area', 'price_doc']]
    TARGET = 'price_doc'
    return df_data[FEATURE], df_data[TARGET]


def _load_test_data(cols):
    df = pandas.read_csv(TEST_DATA_FILE)
    df['timestamp'] = df['timestamp'].apply(lambda x: int(x.replace('-', '')))
    df['product_type'] = df['product_type'].apply(lambda x: x == 'Investment')
    area = pandas.get_dummies(df['sub_area'], prefix='sub_area')
    for col in area:
        df[col] = area[col]
    for col in ['culture_objects_top_25', 'thermal_power_plant_raion', 'incineration_raion', 'oil_chemistry_raion',
                'radiation_raion', 'railroad_terminal_raion', 'big_market_raion', 'nuclear_reactor_raion',
                'detention_facility_raion', 'water_1line', 'big_road1_1line', 'railroad_1line']:
        df[col] = df[col].apply(lambda x: x == 'yes')
    map_ecology = {a: i for i, a in enumerate(['no data', 'poor', 'satisfactory', 'good', 'excellent'])}
    df['ecology'] = df['ecology'].apply(map_ecology.get)
    df_data = df.sort_values(['timestamp', 'sub_area', 'product_type']
                             ).fillna(method='ffill').fillna(method='bfill')

    return df_data


def load_macro_data():
    df = pandas.read_csv(MACRO_DATA_FILE)
    df['timestamp'] = df['timestamp'].apply(lambda x: int(x.replace('-', '')))
    cols = [col for col in df.columns.values
            if col not in df.describe().columns.values]
    for col in cols:
        df[col] = df[col].apply(lambda x: float(x.replace(',', '')) if ',' in str(x) else None)
    df_macro = df.sort_values('timestamp').fillna(method='ffill').fillna(method='bfill')

    return df_macro


def load_data():
    data, label = _load_train_data()
    macro = load_macro_data()

    # df = pandas.merge(data, macro, how='left', on='timestamp').sort_values(
    #    'timestamp').fillna(method='ffill').fillna(method='bfill')
    df = data
    cols = [col for col in df if col not in ('id', 'timestamp')]
    if not os.path.exists('train_data.pkl'):
        with open('train_data.pkl', 'wb') as f:
            pickle.dump((df[cols], label), f, -1)
        return df[cols], label
    else:
        with open('train_data.pkl', 'rb') as f:
            return pickle.load(f)


def load_test_data(cols):
    data = _load_test_data(cols)
    macro = load_macro_data()

    # df = pandas.merge(data, macro, how='left', on='timestamp').sort_values(
    #    'timestamp').fillna(method='ffill').fillna(method='bfill')
    df = data
    for col in cols:
        if col not in df:
            df[col] = numpy.zeros(df.shape[0])
    if not os.path.exists('test_data.pkl'):
        with open('test_data.pkl', 'wb') as f:
            pickle.dump(df[cols], f, -1)
        return df[cols]
    else:
        with open('test_data.pkl', 'rb') as f:
            return pickle.load(f)
