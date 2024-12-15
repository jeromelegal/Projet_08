# custom functions to use them in the different notebooks

# import librairies

import matplotlib.pyplot as plt

from sklearn import set_config
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score


import pandas as pd
import numpy as np
from scipy.stats import kurtosis, iqr, skew
import gc
import time
import re
import json
from contextlib import contextmanager

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

import mlflow
from mlflow import MlflowClient
import mlflow.sklearn
from mlflow.models import infer_signature

# ------------------------------ MLFlow functions ----------------------------
# ----------------------------------------------------------------------------

# mlflow experiment initialization 

experiment_description = (
    "This is the scoring credit tool project for 'Prêt à dépenser'. "
)
experiment_tags = {
    "project_name": "scoring-credit",
    "team": "openclassrooms",
    "project_quarter": "Q4-2024",
    "mlflow.note.content": experiment_description
}


def mlflow_initialize_experiment(experiment_name, 
    experiment_description=None, experiment_tags=None):
    """initialize an experiment in MLFlow with the experiment name"""

    
    client = MlflowClient(tracking_uri="postgresql://mlflowuser:mlflowuser@localhost/mlflowdb")
    experiment = client.get_experiment_by_name(experiment_name)

    # verify if experiment exists
    if experiment is not None:
        print(f"experiment_id: {experiment.experiment_id},\nname: {experiment.name},\ndescription: {experiment.tags.get('mlflow.note.content', 'No description')}")
        return experiment_name
    else:
        if experiment_description and experiment_tags is not None:
            experiment_id = client.create_experiment(
                name=experiment_name, 
                tags=experiment_tags
            )
            print(f"New experiment created : {experiment_id},\nname : {experiment_name},\ndescription : {experiment_description}")
            return experiment_name
        else:
            print('Please add an experiment description and tags.\nmlflow_initialize_experiment(experiment_name, experiment_description=None, experiment_tags=None)')
            return


# function to logging model

def log_model_with_mlflow(model, model_name, params, 
                          metrics, id_experiment, run_name):
    with mlflow.start_run(experiment_id=id_experiment, run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, model_name)


# ------------------------------ soring functions ----------------------------
# ----------------------------------------------------------------------------

# custom score function

def business_score(y_true, y_pred, v=5, w=10):
    """business_score metric to ponderate FP and FN"""
    
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    score_brut = (TN + TP - (v * FP) - (w * FN)) / 100

    # calculate min and max
    total_samples = len(y_true)
    score_min = - ((v * total_samples) + (w * total_samples)) / 100
    score_max = total_samples / 100
    
    # normalization into [0, 1] 
    if score_max == score_min:
        score_normalized = 0  
    else:
        score_normalized = (score_brut - score_min) / (score_max - score_min)
    
    return round(score_normalized, 5)


def business_score_gpu(y_true, y_pred, v=5, w=10):
    """business_score metric to ponderate FP and FN in GPU"""
    
    # convert to cuDF Series
    if not isinstance(y_true, cudf.Series):
        y_true = cudf.Series(y_true)
    if not isinstance(y_pred, cudf.Series):
        y_pred = cudf.Series(y_pred)
        
    # convert in int32
    y_true = y_true.astype('int32')
    y_pred = y_pred.astype('int32')
    
    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    score_brut = (TN + TP - (v * FP) - (w * FN)) / 100

    # min and max scores
    total_samples = len(y_true)
    score_min = -((v * total_samples) + (w * total_samples)) / 100
    score_max = total_samples / 100
    
    # normalize score
    if score_max == score_min:
        score_normalized = 0  
    else:
        score_normalized = (score_brut - score_min) / (score_max - score_min)
    
    # convert in float and round
    score_normalized = float(score_normalized)
    return round(score_normalized, 5)

@contextmanager
def timer(name):
    """Decorator that print the elapsed time and argument."""
    timing = {}
    t0 = time.time()
    yield timing
    timing["elapsed"] = time.time() - t0
    print(f"{name} - done in {timing['elapsed']:.0f}s")

def folds(type_kfold='KFold'):
    """choose KFold or StratifiedKFold"""
    if type_kfold == 'KFold':
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
    elif type_kfold == 'StratifiedKFold':
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        print("Wich type of KFold do you want ?")
        return
    return kf


def complete_scoring(log_param, model, X_train, y_train, X_val, y_val, cv=folds()):
    """Evaluate a model and trace in MLFlow with :
    - Cross Validation (accuracy)
    - Confusion Matrix
    - Business_score
    - AUC score
    - ROC graph
    """
    # prepare MLFlow
    mlflow.set_tracking_uri("postgresql://mlflowuser:mlflowuser@192.168.2.189/mlflowdb")
    mlflow.set_experiment(log_param.get('experiment_name'))
    
    # start experience
    with mlflow.start_run(run_name=log_param.get('run_name')): 
        #print(experiment_id)
        for key, value in log_param.items():
            if key in ['model', 'step']:
                mlflow.set_tag(key, value)
            else:
                if key not in ['experiment_name', 'run_name']:
                    mlflow.log_param(key, value)
    
        scores_list = []
    
        # display pipeline
        set_config(display='diagram')
        display(model) 

        # cross validation
        with timer("Cross validation") as t:
            scores = cross_val_score(model, X_train, y_train, cv=cv, n_jobs=2, error_score='raise')
            print("Accuracy scores for each fold:", scores)
            print("Mean accuracy score:", scores.mean())
            scores_list.append(scores)
        mean_cv = scores.mean()
        std_cv = scores.std()
        # trace cv
        mlflow.log_metric("mean_cv_score", mean_cv)
        mlflow.log_metric("std_cv_score", std_cv)
        mlflow.log_metric("cv_time", t['elapsed'])
    
         # train model
        with timer("Fiting model") as t:
            model.fit(X_train, y_train)
        y_pred = model.predict(X_val)   
        # trace fit
        signature = infer_signature(X_val, y_pred)
        mlflow.log_metric("training_time", t['elapsed'])
        mlflow.sklearn.log_model(
            sk_model = model, 
            artifact_path = log_param['run_name'] + "_model",
            signature=signature
        ) 

        # confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        #print("Matrice de confusion:\n", cm)
        scores_list.append(cm)
        # tarce confusion matrix
        with open("confusion_matrix.json", "w") as f:
            json.dump(cm.tolist(), f)
        mlflow.log_artifact("confusion_matrix.json")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title("Matrice de Confusion")
        plt.savefig("confusion_matrix.png") 
        plt.show()
        mlflow.log_artifact("confusion_matrix.png")

        # business score
        bs = business_score(y_val, y_pred)
        print("Business score:", bs)
        scores_list.append(bs)
        # trace business score
        mlflow.log_metric("business_score", bs)
                   
        # probability predict
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        # AUC
        auc_score = roc_auc_score(y_val, y_pred_proba)
        print("AUC:", auc_score)
        scores_list.append(auc_score)
        # trace AUC
        mlflow.log_metric("auc", auc_score)

        # calculate ROC
        fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)

        # ROC graph
        plt.figure()
        plt.step(fpr, tpr, where='post', label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve (Stair-Step)')
        plt.legend(loc="lower right")
        plt.savefig("roc_curve.png")
        plt.show()
        # trace ROC graph
        mlflow.log_artifact("roc_curve.png")

    return scores_list


# --------------------------- preprocessing function -------------------------
# ----------------------------------------------------------------------------

# One-hot encoding for categorical columns with get_dummies
categorical_columns_list = []
def one_hot_encoder(df, reference_columns=None, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    if reference_columns is not None:
        # Add missing columns in the current DataFrame
        for col in reference_columns:
            if col not in df.columns:
                df[col] = 0    
        # Remove extra columns not present in the reference
        extra_columns = [col for col in df.columns if col not in reference_columns]
        df = df.drop(columns=extra_columns)
    new_columns = [c for c in df.columns if c not in original_columns]
    categorical_columns_list.extend(new_columns)
    return df, new_columns

def drop_application_columns(df):
    """ Drop features based on permutation feature importance. """
    
    drop_list = [
        'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'HOUR_APPR_PROCESS_START',
        'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'FLAG_EMAIL', 'FLAG_PHONE',
        'FLAG_OWN_REALTY', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
        'REG_CITY_NOT_WORK_CITY', 'OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
        'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_YEAR', 
        'COMMONAREA_MODE', 'NONLIVINGAREA_MODE', 'ELEVATORS_MODE', 'NONLIVINGAREA_AVG',
        'FLOORSMIN_MEDI', 'LANDAREA_MODE', 'NONLIVINGAREA_MEDI', 'LIVINGAPARTMENTS_MODE',
        'FLOORSMIN_AVG', 'LANDAREA_AVG', 'FLOORSMIN_MODE', 'LANDAREA_MEDI',
        'COMMONAREA_MEDI', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'BASEMENTAREA_AVG',
        'BASEMENTAREA_MODE', 'NONLIVINGAPARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 
        'LIVINGAPARTMENTS_AVG', 'ELEVATORS_AVG', 'YEARS_BUILD_MEDI', 'ENTRANCES_MODE',
        'NONLIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'LIVINGAPARTMENTS_MEDI',
        'YEARS_BUILD_MODE', 'YEARS_BEGINEXPLUATATION_AVG', 'ELEVATORS_MEDI', 'LIVINGAREA_MEDI',
        'YEARS_BEGINEXPLUATATION_MODE', 'NONLIVINGAPARTMENTS_AVG'
    ]
    # Drop most flag document columns
    for doc_num in [2,4,5,6,7,9,10,11,12,13,14,15,16,17,19,20,21]:
        drop_list.append('FLAG_DOCUMENT_{}'.format(doc_num))
    df.drop(drop_list, axis=1, inplace=True)
    return df

# Preprocess application_train.csv and application_test.csv
def application_train_test(folder='', nan_as_category = False):
    # Read data and merge
    df = pd.read_csv(folder + 'application_train.csv')
    test_df = pd.read_csv(folder + 'application_test.csv')
    df = pd.concat([df, test_df], ignore_index= True)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    del test_df
    gc.collect()
    df = drop_application_columns(df)
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(folder='', nan_as_category = True):
    bureau = pd.read_csv(folder + 'bureau.csv')
    bb = pd.read_csv(folder + 'bureau_balance.csv')
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(folder='', nan_as_category=True):
    prev = pd.read_csv(folder + 'previous_application.csv')
    prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)
    
    # Days 365.243 values -> nan
    days_cols = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION']
    prev[days_cols] = prev[days_cols].replace(365243, np.nan)
    
    # Calculate ratio application/credit 
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT'].replace(0, np.nan)
    
    # Replace inf values per nan
    prev['APP_CREDIT_PERC'].replace([np.inf, -np.inf], np.nan, inplace=True)
    prev['APP_CREDIT_PERC'].fillna(0, inplace=True) 

    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    cat_aggregations = {cat: ['mean'] for cat in cat_cols}

    # Aggregate data
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(folder='', nan_as_category = True):
    pos = pd.read_csv(folder + 'POS_CASH_balance.csv')
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg

def installments_payments(folder='', nan_as_category=True):
    ins = pd.read_csv(folder + 'installments_payments.csv')
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)
    
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT'].replace(0, np.nan)
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    
    # Replace infinities and NaNs
    ins['PAYMENT_PERC'].replace([np.inf, -np.inf], np.nan, inplace=True)
    ins['PAYMENT_PERC'].fillna(0, inplace=True)
    
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    
    # Replace infinities in aggregated data
    ins_agg.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    
    del ins
    gc.collect()
    
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(folder='', nan_as_category = True):
    cc = pd.read_csv(folder + 'credit_card_balance.csv')
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg


def last_cleaning(df):
    object_list = ['CC_NAME_CONTRACT_STATUS_Active_MIN', 'CC_NAME_CONTRACT_STATUS_Active_MAX', 
                   'CC_NAME_CONTRACT_STATUS_Approved_MIN', 'CC_NAME_CONTRACT_STATUS_Approved_MAX', 
                   'CC_NAME_CONTRACT_STATUS_Completed_MIN', 'CC_NAME_CONTRACT_STATUS_Completed_MAX',
                   'CC_NAME_CONTRACT_STATUS_Demand_MIN', 'CC_NAME_CONTRACT_STATUS_Demand_MAX', 
                   'CC_NAME_CONTRACT_STATUS_Refused_MIN', 'CC_NAME_CONTRACT_STATUS_Refused_MAX', 
                   'CC_NAME_CONTRACT_STATUS_Sent proposal_MIN', 'CC_NAME_CONTRACT_STATUS_Sent proposal_MAX',
                  'CC_NAME_CONTRACT_STATUS_Signed_MIN', 'CC_NAME_CONTRACT_STATUS_Signed_MAX', 
                   'CC_NAME_CONTRACT_STATUS_nan_MIN', 'CC_NAME_CONTRACT_STATUS_nan_MAX'
                  ]
    drop_list = ['SK_ID_CURR']
    columns_to_encode = [col for col in object_list if col in df.columns]
    for col in columns_to_encode:
        df[col], _ = pd.factorize(df[col])
    df = df.drop(columns=[col for col in drop_list if col in df.columns])
    # impute Nan by '0'
    df = df.fillna(0)
    return df


def full_preprocessing(folder=''):
    df = application_train_test(folder)
    bureau = bureau_and_balance(folder)
    #print("Bureau df shape:", bureau.shape)
    df = df.join(bureau, how='left', on='SK_ID_CURR')
    del bureau
    gc.collect()
    
    prev = previous_applications(folder)
    #print("Previous applications df shape:", prev.shape)
    df = df.join(prev, how='left', on='SK_ID_CURR')
    del prev
    gc.collect()
    
    pos = pos_cash(folder)
    #print("Pos-cash balance df shape:", pos.shape)
    df = df.join(pos, how='left', on='SK_ID_CURR')
    del pos
    gc.collect()
    
    ins = installments_payments(folder)
    #print("Installments payments df shape:", ins.shape)
    df = df.join(ins, how='left', on='SK_ID_CURR')
    del ins
    gc.collect()
    
    cc = credit_card_balance(folder)
    #print("Credit card balance df shape:", cc.shape)
    df = df.join(cc, how='left', on='SK_ID_CURR')
    del cc
    gc.collect()

    df = last_cleaning(df)  
    
    return df


# Preprocess application_train.csv and application_test.csv
def application_train_test_alone(df, reference_columns=None, nan_as_category = False):
    # Read data and merge
    if df is None:
        raise ValueError("Expected a df, but there's nothing !")
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, reference_columns=reference_columns, nan_as_category=nan_as_category)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df = drop_application_columns(df)
    return df

def application_preprocessing(df):
    df = application_train_test_alone(df)
    df = last_cleaning(df)  
    return df

# ---------------------------- classes for pipeline---------------------------
# ----------------------------------------------------------------------------

class FullPreprocessing(BaseEstimator, TransformerMixin):
    def __init__(self, folder=''):
        self.folder = folder

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None):
        df = full_preprocessing(self.folder)
        return df


class ColumnShaper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X.columns = [re.sub(r'[{}" :,\[\]]', '_', col) for col in X.columns]
        return X

class ModelWithThreshold(BaseEstimator, ClassifierMixin):
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        # Utilise les probabilités pour appliquer le seuil
        probabilities = self.model.predict_proba(X)[:, 1]
        return (probabilities >= self.threshold).astype(int)

    def predict_proba(self, X):
        # Retourne les probabilités du modèle
        return self.model.predict_proba(X)
    

class ApplicationPreprocessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.reference_columns = None
        
    def fit(self, X, y=None):
        """
        Fit the preprocessing pipeline on the training data.
        Captures the reference columns for consistency with test data.
        """
        processed_train = application_preprocessing(X.copy())
        self.reference_columns = processed_train.columns.tolist()
        return self

    def transform(self, X):
        """
        Transform the input data (train or test) using the preprocessing pipeline.
        Aligns the columns to match the reference columns captured during fit.
        """
        # Apply preprocessing with reference columns for consistency
        processed_data = application_preprocessing(X.copy())
        
        # Ensure columns match reference columns
        for col in self.reference_columns:
            if col not in processed_data.columns:
                processed_data[col] = 0  # Add missing columns with default value 0
        
        # Drop extra columns not present in the reference
        extra_columns = [col for col in processed_data.columns if col not in self.reference_columns]
        processed_data = processed_data.drop(columns=extra_columns)

        # Reorder columns to match reference
        processed_data = processed_data[self.reference_columns]
        
        return processed_data

    def transform_to_dataframe(self, X):
        """
        Transform input data and return as a pandas DataFrame.
        """
        processed_array = self.transform(X)
        return pd.DataFrame(processed_array, columns=self.reference_columns)