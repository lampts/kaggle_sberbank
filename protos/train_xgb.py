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

from load_data import load_train_data, load_test_data
CHUNK_SIZE = 100000


def rmse(label, pred):
    return np.sqrt(((pred - label)**2).mean())


def rmsel(label, pred):
    return np.sqrt(((np.log1p(pred) - np.log1p(label))**2).mean())


def rmsel_log(label, pred):
    label = np.exp(label) - 1
    pred = np.exp(pred) - 1
    return np.sqrt(((np.log1p(pred) - np.log1p(label))**2).mean())


def rmsel_metric(pred, dmatrix):
    label = dmatrix.get_label()
    return 'rmsel', rmsel(label, pred)


def rmsel_log_metric(label, pred):
    return 'rmsel', rmsel_log(label, pred)


def rmsel_fobj(labels, preds):
    """
    smoothed_preds = preds + 1
    loss = np.log(smoothed_preds) - np.log(labels + 1)
    n = preds.shape[0]
    grad = - 2 * (loss / smoothed_preds) / n
    hess = - 2 * ((loss - 1) / np.power(smoothed_preds, 2)) / n
    """
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad, hess
    # grad = - 2 * (preds - labels)
    # hess = 2 * preds
    # return grad, hess

if __name__ == '__main__':
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = FileHandler('train.py.log', 'w')
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
    cols = x_train.columns.values
    x_train = x_train.values  # [:, FEATURE]
    y_train = y_train_orig  # np.log1p(y_train_orig)

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

    for params in tqdm(list(ParameterGrid(all_params))):
        cnt = 0
        list_score = []
        list_score2 = []
        list_best_iter = []
        all_pred = np.zeros(y_train.shape[0])
        for train, test in [[cv[:-1000], cv[-1000:]]]:
            trn_x = x_train[train]
            val_x = x_train[test]

            trn_y = y_train[train]
            val_y = y_train[test]
            val_y2 = y_train_orig[test]

            dtrain = xgb.DMatrix(trn_x, trn_y)
            dtest = xgb.DMatrix(val_x, val_y)

            clf = xgb.train(params,
                            dtrain,
                            feval=rmsel_metric,
                            evals=[(dtest, 'val')],
                            num_boost_round=1000,
                            early_stopping_rounds=20)
            pred = clf.predict(dtest)
            all_pred[test] = pred

            _score = rmsel(val_y, pred)
            _score2 = rmse(val_y2, pred)  # np.exp(pred) - 1)  # - roc_auc_score(val_y, pred)
            # logger.debug('   _score: %s' % _score)
            list_score.append(_score)
            list_score2.append(_score2)
            if clf.best_iteration != -1:
                list_best_iter.append(clf.best_iteration)
            else:
                list_best_iter.append(params['n_estimators'])
            break
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
    """
    for params in ParameterGrid(all_params):
        min_params = params
    dtrain = xgb.DMatrix(x_train, y_train)
    cv_output = xgb.cv(params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
                       verbose_eval=50, show_stdv=False)
    """
    # x_train = np.r_[x_train, x_train_rev]
    # y_train = np.r_[y_train, y_train]
    # sample_weight = np.r_[sample_weight, sample_weight]
    clf = xgb.train(min_params,
                    dtrain,
                    num_boost_round=params['n_estimators'])

    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f, -1)
    del x_train
    gc.collect()

    with open('model.pkl', 'rb') as f:
        clf = pickle.load(f)

    x_test = xgb.DMatrix(load_test_data(cols).values)  # [:, FEATURE]

    logger.info('train end')
    preds = []
    p_test = clf.predict(x_test)
    # p_test = np.exp(p_test) - 1
    sub = pd.read_csv('../data/sample_submission.csv')

    sub['price_doc'] = p_test
    sub.to_csv('submit.csv', index=False)
    logger.info('learn start')
