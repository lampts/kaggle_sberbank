import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, cross_val_predict
import xgboost as xgb
from lightgbm.sklearn import LGBMRegressor
from sklearn.metrics import log_loss, roc_auc_score
import gc
from logging import getLogger
logger = getLogger(__name__)
from tqdm import tqdm
from features_tmp import FEATURE
from sklearn.model_selection import TimeSeriesSplit
import math
from load_data import load_train_data2 as load_train_data
from load_data import load_test_data2 as load_test_data
CHUNK_SIZE = 100000

IS_LOG = False
VALID_NUM = 10000  # 3385
LB_NUM = 2682


def rmse(label, pred):
    if IS_LOG:
        label = np.exp(label) - 1
        pred = np.exp(pred) - 1
    return np.sqrt(((pred - label)**2).mean())


def rmsel(label, pred):
    if IS_LOG:
        label = np.exp(label) - 1
        pred = np.exp(pred) - 1
    pred = np.where(pred < 0, 0, pred)
    return np.sqrt(((np.log1p(pred) - np.log1p(label))**2).mean())


def rmsel_metric(pred, dmatrix):
    label = dmatrix.get_label()
    return 'rmsel', rmsel(label, pred)


def tune(x_train, y_train_orig, cols):
    logger.info('{}'.format(cols))
    x_train = x_train[cols].values  # [:, FEATURE]
    if IS_LOG:
        y_train = np.log1p(y_train_orig)
    else:
        y_train = y_train_orig

    logger.info('x_shape: {}'.format(x_train.shape))
    # x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)
    all_params = {
        'eta': [0.05],
        'max_depth': [5],
        'subsample': [0.7],
        'colsample_bytree': [0.7],
        'objective': ['reg:linear'],
        #'eval_metric': [['rmse', rmsel_metric]],
        'silent': [1]
    }
    min_score = (100, 100, 100)
    min_params = None
    use_score = 0
    cv = np.arange(x_train.shape[0])

    for params in list(ParameterGrid(all_params)):
        #cv = TimeSeriesSplit(n_splits=5).split(x_train)
        cnt = 0
        list_score = []
        list_score2 = []
        list_best_iter = []
        all_pred = np.zeros(y_train.shape[0])
        for train, test in [[cv[:-VALID_NUM], cv[-VALID_NUM:]]]:
            trn_x = x_train[train]
            val_x = x_train[test]

            trn_y = y_train[train]
            val_y = y_train[test]

            dtrain = xgb.DMatrix(trn_x, trn_y)
            dtest = xgb.DMatrix(val_x, val_y)

            clf = xgb.train(params,
                            dtrain,
                            feval=rmsel_metric,
                            evals=[(dtest, 'val')],
                            num_boost_round=1000,  # 384,
                            early_stopping_rounds=100)
            pred = clf.predict(dtest)
            all_pred[test] = pred

            _score = rmsel(val_y, pred)
            _score2 = rmse(val_y, pred)  # np.exp(pred) - 1)  # - roc_auc_score(val_y, pred)
            # logger.debug('   _score: %s' % _score)
            list_score.append(_score)
            list_score2.append(_score2)
            if clf.best_iteration != -1:
                list_best_iter.append(clf.best_iteration)
            else:
                list_best_iter.append(params['n_estimators'])

        # with open('tfidf_all_pred2_7.pkl', 'wb') as f:
        #    pickle.dump(all_pred, f, -1)

        logger.info('trees: {}'.format(list_best_iter))
        params['n_estimators'] = np.mean(list_best_iter, dtype=int)
        score = (np.mean(list_score), np.min(list_score), np.max(list_score))
        score2 = (np.mean(list_score2), np.min(list_score2), np.max(list_score2))

        logger.info('param: %s' % (params))
        logger.info('loss: {} (avg min max {})'.format(score[use_score], score))
        logger.info('score: {} (avg min max {})'.format(score2[use_score], score2))
        if min_score[use_score] > score[use_score]:
            min_score = score
            min_score2 = score2
            min_params = params
        logger.info('best score: {} {}'.format(min_score[use_score], min_score))
        logger.info('best score2: {} {}'.format(min_score2[use_score], min_score2))
        logger.info('best_param: {}'.format(min_params))

    gc.collect()
    return min_score[use_score]

if __name__ == '__main__':
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = FileHandler('col_tune.log', 'w')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.setLevel('INFO')
    logger.addHandler(handler)

    logger.info('load start')

    x_train, y_train_orig = load_train_data()
    all_cols = x_train.columns.values.tolist()
    # 0.41964659008313887 :: 10000
    cols = ['full_sq', 'life_sq', 'floor', 'max_floor', 'material', 'build_year', 'kitch_sq', 'product_type', 'area_m', 'raion_popul', 'green_zone_part', 'indust_part', 'children_preschool', 'children_school', 'school_quota', 'culture_objects_top_25',
            'ID_metro', 'park_km', 'industrial_km', 'water_treatment_km', 'ID_railroad_station_walk', 'water_1line', 'mkad_km', 'bulvar_ring_km', 'ID_big_road1', 'cafe_avg_price_500', 'cafe_count_500_price_1500', 'cafe_count_2000', 'cafe_count_3000']

    # 0.40232381770340664 :: 5000
    """
    cols = ['full_sq', 'life_sq', 'floor', 'max_floor', 'material', 'build_year', 'kitch_sq', 'product_type', 'area_m', 'raion_popul', 'green_zone_part', 'indust_part', 'children_preschool', 'children_school', 'school_quota', 'culture_objects_top_25',
            'ID_metro', 'park_km', 'industrial_km', 'water_treatment_km', 'ID_railroad_station_walk', 'water_1line', 'mkad_km', 'bulvar_ring_km', 'ID_big_road1', 'cafe_avg_price_500', 'cafe_count_500_price_1500', 'cafe_count_2000', 'cafe_count_3000',
            'work_female', 'ekder_female', '0_13_female', 'raion_build_count_with_material_info', 'ttk_km']
    """
    min_score = tune(x_train, y_train_orig, cols)
    for col in tqdm(all_cols):
        if col in cols:
            continue
        score = tune(x_train, y_train_orig, cols + [col])
        if score < min_score:
            min_score = score
            cols.append(col)
        logger.info('best score: {}'.format(min_score))
        logger.info('best feat: {}'.format(cols))
