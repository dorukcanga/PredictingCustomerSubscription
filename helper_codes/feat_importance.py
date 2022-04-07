import pandas as pd
import numpy as np
import random
import time
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from functools import reduce

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer

from category_encoders.target_encoder import TargetEncoder

import lightgbm as lgb
import xgboost as xgb

import eli5

import scipy.stats as ss
from scipy.stats import pearsonr

from boruta import BorutaPy

from statsmodels.stats.outliers_influence import variance_inflation_factor

from helper_codes.model_wrappers import LGBMWrapper, XGBWrapper, RFWrapper, ANNWrapper

import warnings
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None



def construct_preprocessing_pipeline(preprocessing_dict=None, identity_features=None):
    
    pre_transformers = []
    
    if preprocessing_dict != None:
        for key, value in preprocessing_dict.items():
            pipeline_list = []
            for i in range(1, len(value)):
                step = value[i][0]

                if step == 'imputer':
                    if value[i][1] != 'knn':
                        if len(value[i]) == 2:
                            pipeline_list.append((step, SimpleImputer(strategy=value[i][1])))
                        else:
                            pipeline_list.append((step, SimpleImputer(strategy=value[i][1], fill_value=value[i][2])))
                    else:
                        pipeline_list.append((step, KNNImputer(n_neighbors=value[i][2]))) 

                if step == 'scaler':
                    if value[i][1] == 'minmax':
                        pipeline_list.append((step, MinMaxScaler()))
                    elif value[i][1] == 'standard':
                        pipeline_list.append((step, StandardScaler()))

                if step == 'pca':
                        pipeline_list.append((step, PCA(n_components=value[i][1])))

                if step == 'encoder':
                        if value[i][1] == 'ohe':
                            pipeline_list.append((step, OneHotEncoder(sparse = True, handle_unknown = "ignore")))
                        elif value[i][1] == 'te':
                            if len(value[i]) == 2:
                                pipeline_list.append((step, TargetEncoder(handle_missing='value', handle_unknown='value', min_samples_leaf=1, smoothing=1.0)))
                            elif len(value[i]) == 3:
                                pipeline_list.append((step, TargetEncoder(handle_missing='value', handle_unknown='value', **value[i][2])))

            transformer = Pipeline(steps=pipeline_list)
            pre_transformers.append((key, transformer, value[0]))
                                                 
    if identity_features != None:
        pipeline_list = []
        pipeline_list.append(('identity', FunctionTransformer(func = None)))
                                                     
        transformer = Pipeline(steps=pipeline_list) 
        pre_transformers.append(('identity', transformer, identity_features))
                                                     
    preprocessor = ColumnTransformer(transformers = pre_transformers, remainder='drop')
                                                     
    return preprocessor



class AutoFeatureImportance(object):
    
    
    def __init__(self, modelling_type, data, response_column, numeric_columns=None, categorical_columns=None, weight_column=None,
                 data_sampling=None, data_split=False, split_stratify=None, seed=None):
    
        self.modelling_type = modelling_type
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.seed = seed

        #Data Sampling
        if data_sampling and data_sampling <= 1:
            self.data = data.sample(frac=data_sampling, random_state=seed)
        elif data_sampling and data_sampling > 1:
            self.data = data.sample(n=data_sampling, random_state=seed)
        else:
            self.data = data

        #Data Spliting
        if data_split == 'auto' and weight_column == None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data[numeric_columns+categorical_columns],
                                                                                    self.data[response_column],
                                                                                    test_size=0.2, random_state=self.seed, stratify=split_stratify)
            self.weight_train, self.weight_test = None, None
            
        elif data_split == 'auto' and weight_column != None:
            self.X_train, self.X_test, self.y_train, self.y_test, self.weight_train, self.weight_test = train_test_split(self.data[numeric_columns+categorical_columns],
                                                                                                                         self.data[response_column], self.data[weight_column],
                                                                                                                         test_size=0.2, random_state=self.seed,
                                                                                                                         stratify=split_stratify)
        elif type(data_split) == list:
            self.X_train = self.data.loc[self.data[data_split[0]] == data_split[1], numeric_columns+categorical_columns]
            self.X_test = self.data.loc[self.data[data_split[0]] == data_split[2], numeric_columns+categorical_columns]
            self.y_train = self.data.loc[self.data[data_split[0]] == data_split[1], response_column]
            self.y_test = self.data.loc[self.data[data_split[0]] == data_split[2], response_column]
            
            self.weight_train, self.weight_test = None, None

            if weight_column != None:
                self.weight_train = self.data.loc[self.data[data_split[0]] == data_split[1], weight_column]
                self.weight_test = self.data.loc[self.data[data_split[0]] == data_split[2], weight_column]

        else:
            self.X_train, self.y_train = self.data[numeric_columns+categorical_columns], self.data[response_column]
            self.weight_train, self.weight_test = None, None

            if weight_column != None:
                self.weight_train = self.data[weight]
                
                
                
                
    def lgbm_importance(self, hyperparam_dict=None, return_preds=False, return_plot=False):

        if hyperparam_dict == None:
            hyperparam_dict = {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'num_leaves': 20,
                               'random_state':self.seed, 'n_jobs':4, 'importance_type':'gain'}

        #Model Training   
        model = LGBMWrapper(modelling_type=self.modelling_type, hyperparams=hyperparam_dict)
        model.fit(X_train=self.X_train, y_train=self.y_train, sample_weight = self.weight_train)

        if return_preds:
            y_preds = model.predict(self.X_test)

        #Feature Importance DataFrame
        self.lgbm_importance_df = pd.DataFrame({
            'feature' : self.X_train.columns,
            'importance' : model.feature_importances_
        }).sort_values('importance', ascending=False)

        #Importance Plot
        if return_plot:
            plt.figure(figsize=(20, 10))
            sns.barplot(x="importance", y="feature", data=self.lgbm_importance_df)
            plt.tight_layout()
            plt.show()

        if return_preds:
            return self.lgbm_importance_df, y_preds
        else:
            return self.lgbm_importance_df
        
        
    def xgb_importance(self, hyperparam_dict=None, return_preds=False, return_plot=False):

        if hyperparam_dict == None:
            hyperparam_dict = {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200,
                               'random_state':self.seed, 'n_jobs':4, 'importance_type':'gain'}

        #Model Training
        preprocessing_dict = {
                'numeric_trans' : [self.numeric_columns, ('imputer', 'mean')],
                'categorical_trans' : [self.categorical_columns, ('encoder', 'te')]
            }
        
        preprocessor = construct_preprocessing_pipeline(preprocessing_dict=preprocessing_dict, identity_features=None)
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model_object', XGBWrapper(modelling_type=self.modelling_type, hyperparams=hyperparam_dict))])
        model.fit(X=self.X_train, y=self.y_train, model_object__sample_weight = self.weight_train)

        if return_preds:
            y_preds = model.predict(self.X_test)

        #Feature Importance DataFrame
        self.xgb_importance_df = pd.DataFrame({
            'feature' : self.X_train.columns,
            'importance' : model.named_steps['model_object'].feature_importances_
        }).sort_values('importance', ascending=False)

        #Importance Plot
        if return_plot:
            plt.figure(figsize=(20, 10))
            sns.barplot(x="importance", y="feature", data=self.xgb_importance_df)
            plt.tight_layout()
            plt.show()

        if return_preds:
            return self.xgb_importance_df, y_preds
        else:
            return self.xgb_importance_df
        
        
    def rf_importance(self, hyperparam_dict=None, return_preds=False, return_plot=False):

        if hyperparam_dict == None:
            hyperparam_dict = {'max_depth': 5, 'n_estimators': 200, 'random_state':self.seed, 'n_jobs':4}

        #Model Training
        preprocessing_dict = {
                'numeric_trans' : [self.numeric_columns, ('imputer', 'mean')],
                'categorical_trans' : [self.categorical_columns, ('encoder', 'te')]
            }
        
        preprocessor = construct_preprocessing_pipeline(preprocessing_dict=preprocessing_dict, identity_features=None)
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model_object', RFWrapper(modelling_type=self.modelling_type, hyperparams=hyperparam_dict))])
        model.fit(X=self.X_train, y=self.y_train, model_object__sample_weight = self.weight_train)

        if return_preds:
            y_preds = model.predict(self.X_test)

        #Feature Importance DataFrame
        self.rf_importance_df = pd.DataFrame({
            'feature' : self.X_train.columns,
            'importance' : model.named_steps['model_object'].feature_importances_
        }).sort_values('importance', ascending=False)

        #Importance Plot
        if return_plot:
            plt.figure(figsize=(20, 10))
            sns.barplot(x="importance", y="feature", data=self.rf_importance_df)
            plt.tight_layout()
            plt.show()

        if return_preds:
            return self.rf_importance_df, y_preds
        else:
            return self.rf_importance_df
        

    def lassocv_importance(self, hyperparam_dict=None, return_preds=False, return_plot=False):

        if hyperparam_dict == None:
            hyperparam_dict = {'cv':5, 'random_state':self.seed, 'n_jobs':4}

        #Model Training
        preprocessing_dict = {
                'numeric_trans' : [self.numeric_columns, ('imputer', 'mean')],
                'categorical_trans' : [self.categorical_columns, ('imputer', 'constant', 'unknown'), ('encoder', 'ohe')]
            }

        preprocessor = construct_preprocessing_pipeline(preprocessing_dict=preprocessing_dict, identity_features=None)
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model_object', LassoCV(**hyperparam_dict))])
        model.fit(X=self.X_train, y=self.y_train) # model_object__sample_weight = self.weight_train
        
        #Feature Importance DataFrame
        feature_names = self.numeric_columns + model.named_steps['preprocessor']\
                                                        .named_transformers_["categorical_trans"]['encoder']\
                                                        .get_feature_names(self.categorical_columns).tolist()

        self.lasso_importance_df = pd.DataFrame({'feature':feature_names, 'importance':model['model_object'].coef_})

        self.lasso_importance_df['importance'] = abs(self.lasso_importance_df['importance'])

        mapping_dict={}
        for i in self.lasso_importance_df.feature.values[len(self.numeric_columns):]:
            temp_mapping_list = [x for x in self.categorical_columns if x in i]
            mapping_dict[i] = max(temp_mapping_list, key=len)

        for i in self.lasso_importance_df.feature.values[:len(self.numeric_columns)]:
            mapping_dict[i] = i

        self.lasso_importance_df['org_feature'] = [mapping_dict[x] for x in self.lasso_importance_df.feature]

        self.lasso_importance_df = self.lasso_importance_df.groupby('org_feature').importance.sum().to_frame()
        self.lasso_importance_df.reset_index(inplace=True, drop=False)
        self.lasso_importance_df.columns = ['feature', 'importance']
        self.lasso_importance_df.sort_values('importance', ascending=False, inplace=True)
        
        #Importance Plot
        if return_plot:
            plt.figure(figsize=(20, 10))
            sns.barplot(x="importance", y="feature", data=self.lasso_importance_df)
            plt.tight_layout()
            plt.show()

        return self.lasso_importance_df
    

    def eli5_importance(self, lgbm_hyperparams=None, perimp_hyperparams=None, return_plot=False):

        if lgbm_hyperparams == None:
            lgbm_hyperparams = {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'num_leaves': 20,
                               'random_state':self.seed, 'n_jobs':4, 'importance_type':'gain'}

        if perimp_hyperparams == None:
            perimp_hyperparams = {'cv':5, 'random_state':self.seed}

        #Target Encoded Dataset
        
        self.X_train_te = self.X_train.copy()

        # Missing Value Imputation
        for col_name in self.X_train_te.columns:
            na_count = self.X_train_te[col_name].isna().sum()
            if na_count > 0 and col_name in self.numeric_columns:
                self.X_train_te[col_name] = self.X_train_te[col_name].fillna(value=self.X_train_te[col_name].mean())
            elif na_count > 0 and col_name in self.categorical_columns:
                self.X_train_te[col_name] = self.X_train_te[col_name].fillna(value=self.X_train_te[col_name].mode().values[0])
        
        te = TargetEncoder(handle_missing='value', handle_unknown='value', min_samples_leaf=1, smoothing=1.0)
        X_train_cat = te.fit_transform(self.X_train_te[self.categorical_columns], y=self.y_train)

        self.X_train_te = pd.concat([self.X_train_te[self.numeric_columns],X_train_cat], axis=1)

        for col_name in self.numeric_columns+self.categorical_columns:
            self.X_train_te[col_name] = self.X_train_te[col_name].astype('float64')
            
        self.X_train_te.reset_index(inplace=True, drop=True)
        self.y_train_te = self.y_train.reset_index(drop=True)

        #Permutation Importance Training
        model = LGBMWrapper(modelling_type=self.modelling_type, hyperparams=lgbm_hyperparams, return_probability=False)

        perm = eli5.sklearn.PermutationImportance(model.model, **perimp_hyperparams)
        perm.fit(self.X_train_te[self.numeric_columns + self.categorical_columns], self.y_train_te)

        self.perimp_importance_df = pd.DataFrame([list(self.X_train_te.columns),list(abs(perm.feature_importances_))]).T
        self.perimp_importance_df.columns = ['feature', 'importance']
        self.perimp_importance_df = self.perimp_importance_df.sort_values('importance', ascending=False)

        #Importance Plot
        if return_plot:
            plt.figure(figsize=(20, 10))
            sns.barplot(x="importance", y="feature", data=self.perimp_importance_df)
            plt.tight_layout()
            plt.show()
            
        return self.perimp_importance_df
    
    
    
    def rfecv_importance(self, lgbm_hyperparams=None, rfecv_hyperparams=None, return_plot=False):

        if lgbm_hyperparams == None:
            lgbm_hyperparams = {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'num_leaves': 20,
                               'random_state':self.seed, 'n_jobs':4, 'importance_type':'gain'}

        if rfecv_hyperparams == None:
            rfecv_hyperparams = {'step':1, 'min_features_to_select':1, 'cv':5}

        #Target Encoded Dataset
        
        self.X_train_te = self.X_train.copy()

        # Missing Value Imputation
        for col_name in self.X_train_te.columns:
            na_count = self.X_train_te[col_name].isna().sum()
            if na_count > 0 and col_name in self.numeric_columns:
                self.X_train_te[col_name] = self.X_train_te[col_name].fillna(value=self.X_train_te[col_name].mean())
            elif na_count > 0 and col_name in self.categorical_columns:
                self.X_train_te[col_name] = self.X_train_te[col_name].fillna(value=self.X_train_te[col_name].mode().values[0])
        
        te = TargetEncoder(handle_missing='value', handle_unknown='value', min_samples_leaf=1, smoothing=1.0)
        X_train_cat = te.fit_transform(self.X_train_te[self.categorical_columns], y=self.y_train)

        self.X_train_te = pd.concat([self.X_train_te[self.numeric_columns],X_train_cat], axis=1)

        for col_name in self.numeric_columns+self.categorical_columns:
            self.X_train_te[col_name] = self.X_train_te[col_name].astype('float64')
            
        self.X_train_te.reset_index(inplace=True, drop=True)
        self.y_train_te = self.y_train.reset_index(drop=True)

        #Permutation Importance Training
        model = LGBMWrapper(modelling_type=self.modelling_type, hyperparams=lgbm_hyperparams, return_probability=False)

        rfe = RFECV(model.model, **rfecv_hyperparams)
        rfe.fit(self.X_train_te[self.numeric_columns + self.categorical_columns], self.y_train_te)

        self.rfe_importance_df = pd.DataFrame([list(self.X_train_te.columns),list(abs(rfe.ranking_))]).T
        self.rfe_importance_df.columns = ['feature', 'importance']
        self.rfe_importance_df = self.rfe_importance_df.sort_values('importance', ascending=True)

        #Importance Plot
        if return_plot:
            plt.figure(figsize=(20, 10))
            sns.barplot(x="importance", y="feature", data=self.rfe_importance_df)
            plt.tight_layout()
            plt.show()
            
        return self.rfe_importance_df
    
    
    def pearson_correlation(self, corr_threshold=0.9, print_results=False):
        """
        Calculates correlation between elements of the given feature list and returns correlation dataframe.
        """
        result_dict = {'i':[], 'j':[], 'correlation_score':[]}
        tuple_list = []
        for i in self.numeric_columns:
            for j in self.numeric_columns:
                if i != j and (j,i) not in tuple_list:
                    tuple_list.append((i,j))
                    corr = abs(pearsonr(self.X_train[i].fillna(self.X_train[i].mean()), self.X_train[j].fillna(self.X_train[j].mean()))[0])

                    result_dict['i'].append(i)
                    result_dict['j'].append(j)
                    result_dict['correlation_score'].append(corr)

                    if corr >= corr_threshold and print_results:
                        print(i,j,corr)

        self.pearson_corr_df = pd.DataFrame(result_dict)

        return self.pearson_corr_df
    
    
    def cramers_v_correlation(self, corr_threshold=0.9, print_results=False):
        """
        Calculates cramers_v score between elements of the given feature list and returns score dataframe.
        """

        def cramers_v(x, y):
            confusion_matrix = pd.crosstab(x,y)
            chi2 = ss.chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            phi2 = chi2/n
            r,k = confusion_matrix.shape
            phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
            rcorr = r-((r-1)**2)/(n-1)
            kcorr = k-((k-1)**2)/(n-1)
            return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))

        result_dict = {'i':[], 'j':[], 'correlation_score':[]}
        tuple_list = []
        for i in self.categorical_columns:
            for j in self.categorical_columns:
                if i != j and (j,i) not in tuple_list:
                    tuple_list.append((i,j))
                    value_count_i, value_count_j = self.X_train[i].value_counts(normalize=True)[0], self.X_train[j].value_counts(normalize=True)[0]
                    if value_count_i != 1 and value_count_j != 1:
                        cramers_v_score = cramers_v(self.X_train[i], self.X_train[j])

                        result_dict['i'].append(i)
                        result_dict['j'].append(j)
                        result_dict['correlation_score'].append(cramers_v_score)

                        if cramers_v_score > corr_threshold and print_results:
                            print(i,j,cramers_v_score)

        self.cramers_v_df = pd.DataFrame(result_dict)

        return self.cramers_v_df
    
    
    def boruta_importance(self, lgbm_hyperparams=None, boruta_hyperparams=None):

        if lgbm_hyperparams == None:
            lgbm_hyperparams = {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'num_leaves': 20,
                               'random_state':self.seed, 'n_jobs':4, 'importance_type':'gain'}

        if boruta_hyperparams == None:
            boruta_hyperparams = {'n_estimators':'auto', 'random_state':self.seed}

        #Target Encoded Dataset
        
        self.X_train_te = self.X_train.copy()

        # Missing Value Imputation
        for col_name in self.X_train_te.columns:
            na_count = self.X_train_te[col_name].isna().sum()
            if na_count > 0 and col_name in self.numeric_columns:
                self.X_train_te[col_name] = self.X_train_te[col_name].fillna(value=self.X_train_te[col_name].mean())
            elif na_count > 0 and col_name in self.categorical_columns:
                self.X_train_te[col_name] = self.X_train_te[col_name].fillna(value=self.X_train_te[col_name].mode().values[0])
        
        te = TargetEncoder(handle_missing='value', handle_unknown='value', min_samples_leaf=1, smoothing=1.0)
        X_train_cat = te.fit_transform(self.X_train_te[self.categorical_columns], y=self.y_train)

        self.X_train_te = pd.concat([self.X_train_te[self.numeric_columns],X_train_cat], axis=1)

        for col_name in self.numeric_columns+self.categorical_columns:
            self.X_train_te[col_name] = self.X_train_te[col_name].astype('float64')
            
        self.X_train_te.reset_index(inplace=True, drop=True)
        self.y_train_te = self.y_train.reset_index(drop=True)

        #Permutation Importance Training
        model = LGBMWrapper(modelling_type=self.modelling_type, hyperparams=lgbm_hyperparams, return_probability=False)

        boruta_imp = BorutaPy(model.model, **boruta_hyperparams)
        boruta_imp = boruta_imp.fit(self.X_train_te.values, self.y_train_te.values)
        
        self.boruta_df = pd.DataFrame({'feature' : self.X_train_te.columns, 'boruta_ranking' : boruta_imp.ranking_,
                                  'boruta_support' : boruta_imp.support_, 'boruta_weak_support': boruta_imp.support_weak_})

        self.boruta_df.sort_values('boruta_ranking', ascending=True, inplace=True)
            
        return self.boruta_df
    
    def vif(self):

        #Target Encoded Dataset
        
        self.X_train_te = self.X_train.copy()

        # Missing Value Imputation
        for col_name in self.X_train_te.columns:
            na_count = self.X_train_te[col_name].isna().sum()
            if na_count > 0 and col_name in self.numeric_columns:
                self.X_train_te[col_name] = self.X_train_te[col_name].fillna(value=self.X_train_te[col_name].mean())
            elif na_count > 0 and col_name in self.categorical_columns:
                self.X_train_te[col_name] = self.X_train_te[col_name].fillna(value=self.X_train_te[col_name].mode().values[0])
        
        te = TargetEncoder(handle_missing='value', handle_unknown='value', min_samples_leaf=1, smoothing=1.0)
        X_train_cat = te.fit_transform(self.X_train_te[self.categorical_columns], y=self.y_train)

        self.X_train_te = pd.concat([self.X_train_te[self.numeric_columns],X_train_cat], axis=1)

        for col_name in self.numeric_columns+self.categorical_columns:
            self.X_train_te[col_name] = self.X_train_te[col_name].astype('float64')
            
        self.X_train_te.reset_index(inplace=True, drop=True)
        self.y_train_te = self.y_train.reset_index(drop=True)
        
        self.X_train_te = self.X_train_te.assign(const=1)

        self.vif_df = pd.DataFrame([variance_inflation_factor(self.X_train_te.values, i) 
                       for i in range(self.X_train_te.shape[1])], 
                      index=self.X_train_te.columns).reset_index()

        self.vif_df.columns = ['feature', 'vif_score']
        
        return self.vif_df
    
    
    def auto_drop_corr_features(self, check_vif = False, return_removed_features = False, keep_high_cardinal_cat_feature=False):

        if hasattr(self, 'pearson_corr_df') == False:
            corr_df = self.pearson_correlation()
        if hasattr(self, 'cramers_v_df') == False:
            cramers_df = self.cramers_v_correlation()
        if hasattr(self, 'vif_df') == False and check_vif == True:
            vif_df = self.vif()

        #Pearson Corr
        corr_df2 = self.pearson_corr_df[self.pearson_corr_df.correlation_score >= 0.9]

        counter_dict = Counter(corr_df2.i.tolist() + corr_df2.j.tolist())
        counter_dict = {k: v for k, v in sorted(counter_dict.items(), key=lambda item: item[1], reverse=True)}

        corr_features_dict = {}
        kept_features, removed_features = [], []
        for col_name in counter_dict.keys():
            if col_name not in removed_features:
                temp_df = corr_df2[(corr_df2.i == col_name) | (corr_df2.j == col_name)]
                temp_removed_list = [x for x in set(temp_df.i.tolist() + temp_df.j.tolist()) if x != col_name]

                kept_features.append(col_name)
                removed_features = removed_features + [x for x in temp_removed_list if x not in removed_features]
                corr_features_dict[col_name] = temp_removed_list

        self.num_kept_features, self.num_removed_features = kept_features, removed_features

        #Cramers V
        cramers_df2 = self.cramers_v_df[self.cramers_v_df.correlation_score >= 0.9]

        counter_dict = Counter(cramers_df2.i.tolist() + cramers_df2.j.tolist())
        counter_dict = {k: v for k, v in sorted(counter_dict.items(), key=lambda item: item[1], reverse=True)}

        cramers_features_dict = {}
        kept_features, removed_features = [], []

        for col_name in counter_dict.keys():
            if col_name not in removed_features:
                temp_df = cramers_df2[(cramers_df2.i == col_name) | (cramers_df2.j == col_name)]
                temp_removed_list = [x for x in set(temp_df.i.tolist() + temp_df.j.tolist()) if x != col_name]

                if keep_high_cardinal_cat_feature:
                    temp_removed_list = [i for i in temp_removed_list if model_data[i].nunique() / model_data[col_name].nunique() < 1]
                else:
                    temp_removed_list = [i for i in temp_removed_list if 0.75 <= model_data[i].nunique() / model_data[col_name].nunique() <= 1.25]

                kept_features.append(col_name)
                removed_features = removed_features + [x for x in temp_removed_list if x not in removed_features]
                cramers_features_dict[col_name] = temp_removed_list

        self.cat_kept_features, self.cat_removed_features = kept_features, removed_features

        self.drop_column(column_list = self.num_removed_features + self.cat_removed_features)

        if return_removed_features:
            return self.num_removed_features + self.cat_removed_features
        

        
    def auto_select(self, importance_types=['lgbm', 'rf', 'perimp'], remove_correlated_features=False, remove_multicollinearity=False,
                    lgbm_hyperparams=None, xgb_hyperparams=None, rf_hyperparams=None,
                    lassocv_hyperparams=None, perimp_hyperparams=None, rfecv_hyperparams=None, boruta_hyperparams=None):

        if remove_correlated_features:
            self.auto_drop_corr_features(check_vif = remove_multicollinearity, return_removed_features = False, keep_high_cardinal_cat_feature=False)
            print("Correlated Features are removed")


        dataframes, column_names = [], ['feature']
        if 'lgbm' in importance_types:
            try:
                lgbm_importance_df = self.lgbm_importance(hyperparam_dict=lgbm_hyperparams)
                dataframes.append(lgbm_importance_df)
                column_names.append('lgbm_importance')
                print("LightGBM Feature Importance is finished")
            except Exception:
                print("Error in LightGBM Feature Importance")

        if 'xgb' in importance_types:
            try:
                xgb_importance_df = self.xgb_importance(hyperparam_dict=xgb_hyperparams)
                dataframes.append(xgb_importance_df)
                column_names.append('xgb_importance')
                print("XGBoost Feature Importance is finished")
            except Exception:
                print("Error in XGBoost Feature Importance")

        if 'rf' in importance_types:
            try:
                rf_importance_df = self.rf_importance(hyperparam_dict=rf_hyperparams)
                dataframes.append(rf_importance_df)
                column_names.append('rf_importance')
                print("Random Forest Feature Importance is finished")
            except Exception:
                print("Error in Random Forest Feature Importance")

        if 'lassocv' in importance_types:
            try:
                lassocv_importance_df = self.lassocv_importance(hyperparam_dict=lassocv_hyperparams)
                dataframes.append(lassocv_importance_df)
                column_names.append('lassocv_coefs')
                print("LassoCV Feature Importance is finished")
            except Exception:
                print("Error in LassoCV Feature Importance")

        if 'perimp' in importance_types:
            try:
                perimp_importance_df = self.eli5_importance(lgbm_hyperparams=lgbm_hyperparams, perimp_hyperparams=perimp_hyperparams)
                dataframes.append(perimp_importance_df)
                column_names.append('permutation_importance')
                print("ELI5 Permutation Importance Feature Importance is finished")
            except Exception:
                print("Error in ELI5 Permutation Importance Feature Importance")

        if 'rfecv' in importance_types:
            try:
                rfecv_importance_df = self.rfecv_importance(lgbm_hyperparams=lgbm_hyperparams, rfecv_hyperparams=rfecv_hyperparams)
                dataframes.append(rfecv_importance_df)
                column_names.append('rfecv_rankings')
                print("RFECV Feature Importance is finished")
            except Exception:
                print("Error in RFECV Feature Importance")

        if 'boruta' in importance_types:
            try:
                boruta_df = self.boruta_importance(lgbm_hyperparams=lgbm_hyperparams, boruta_hyperparams=boruta_hyperparams)
                dataframes.append(boruta_df)
                column_names = column_names + boruta_df.columns.tolist()[1:]
                print("Boruta Feature Importance is finished")
            except Exception:
                print("Error in Boruta Feature Importance")


        self.final_importance_df = reduce(lambda  left,right: pd.merge(left,right,on=['feature'], how='outer'),
                                        dataframes)

        self.final_importance_df.columns = column_names

        return self.final_importance_df
    
    
    def drop_column(self, column_list):
        self.X_train.drop(column_list, axis=1, inplace=True)
        self.X_test.drop(column_list, axis=1, inplace=True)
        
        self.categorical_columns = [i for i in self.categorical_columns if i not in column_list]
        self.numeric_columns = [i for i in self.numeric_columns if i not in column_list]