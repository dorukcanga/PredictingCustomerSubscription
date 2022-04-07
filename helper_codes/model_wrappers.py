import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import tensorflow as tf

class LGBMWrapper(object):
    """
    A wrapper for lightgbm model with fit, predict and predict_proba methods.
    """    
    
    def __init__(self, modelling_type, hyperparams, return_probability=False):
        self.return_probability = return_probability 
        if modelling_type == 'classification':
            self.model = lgb.LGBMClassifier(**hyperparams)
        elif modelling_type == 'regression':
            self.model = lgb.LGBMRegressor(**hyperparams)
            
    def set_params(self, hyperparams):
        self.model = self.model.set_params(**hyperparams)
            
    def fit(self, X_train, y_train, fit_params={'verbose':0},\
            X_valid=None, y_valid=None, evaluation_metric=None, sample_weight=None, eval_sample_weight=None):
        
        eval_set, eval_names = None, None
        if X_valid is not None:
            eval_set = [(X_valid, y_valid)]
            eval_names = ['valid']
            
        self.model.fit(X=X_train, y=y_train, eval_set=eval_set, eval_names=eval_names,\
                       eval_metric=evaluation_metric, sample_weight=sample_weight, eval_sample_weight=eval_sample_weight, **fit_params)
        
        
        self.feature_importances_ = self.model.feature_importances_
        
    def predict(self, X_test):
        if self.return_probability:
            return self.model.predict_proba(X_test)
        else:
            return self.model.predict(X_test)
        

class XGBWrapper(object):
    """
    A wrapper for xgboost model with fit, predict and predict_proba methods.
    """
    
    def __init__(self, modelling_type, hyperparams, return_probability=False):
        self.return_probability = return_probability
        if modelling_type == 'classification':
            self.model = xgb.XGBClassifier(**hyperparams)
        elif modelling_type == 'regression':
            self.model = xgb.XGBRegressor(**hyperparams)
            
    def set_params(self, params):
        self.model = self.model.set_params(**hyperparams)
            
    def fit(self, X_train, y_train, fit_params={'verbose':0},\
            X_valid=None, y_valid=None, evaluation_metric=None, sample_weight=None, eval_sample_weight=None):
        
        eval_set = None
        if X_valid is not None:
            eval_set = [(X_valid, y_valid)]
            
        self.model.fit(X=X_train, y=y_train, eval_set=eval_set,\
                       eval_metric=evaluation_metric, sample_weight=sample_weight, sample_weight_eval_set=eval_sample_weight, **fit_params)
        
        
        self.feature_importances_ = self.model.feature_importances_
        
    def predict(self, X_test):
        if self.return_probability:
            return self.model.predict_proba(X_test)
        else:
            return self.model.predict(X_test)



class RFWrapper(object):
    """
    A wrapper for random forest model with fit, predict and methods.
    """
    
    def __init__(self, modelling_type, hyperparams, return_probability=False):
        self.return_probability = return_probability
        if modelling_type == 'classification':
            self.model = RandomForestClassifier(**hyperparams)
        elif modelling_type == 'regression':
            self.model = RandomForestRegressor(**hyperparams)
            
    def fit(self, X_train, y_train, sample_weight=None):
            
        self.model.fit(X=X_train, y=y_train, sample_weight=sample_weight)
        
        self.feature_importances_ = self.model.feature_importances_
        
    def predict(self, X_test):
        if self.return_probability:
            return self.model.predict_proba(X_test)
        else:
            return self.model.predict(X_test)


class ANNWrapper_old(object):
    # This class is tensorflow based wrapper object that crated to work with scikit
    # Needs tensorflow to be imported. (import tensorflow as tf)
    # Tensorflow version: 2.4.0
    # Python version: 3.6.9
    # For more detailed information about tensorflow: https://www.tensorflow.org/api_docs/python/tf/all_symbols
    #                                                 https://www.tensorflow.org/api_docs/python/tf/keras/Model
    
    def __init__(self, layers, input_shape, loss, metrics, model_type, epochs, learning_rate=0.001, callbacks=None, validation_data=None,
                 optimizer='adam', out_activation='softmax', layer_activation='relu', batch_size=None, return_probability=False, seed=24):
        # @param layers: List of tuple that carries information about networks layers. Each tuple must covers at least 2 elements.
                # Ex. [(32, 0.2, 'L1', 0.01), (64, 0, 'L1L2', 0.01, 0.02) ]
                # First element represents Dense node count in layer
                # Second elemen represents dropout rate - float [0,1)
                # (Optional) Third element represents kernel regularizer type. L1, L2 or L1L2
                # (Not Optional if Regularizer Type given) Fourth (Fifth) element represents regularization factor.  
        # @param input_shape: Shape of input tensor.
        # @param loss: Loss function's name (see: https://www.tensorflow.org/api_docs/python/tf/keras/losses) or loss function.
        # @param metrics: List of metrics to be evaluated by the model during training and testing. 
        #                 Can be function's name (see: https://www.tensorflow.org/api_docs/python/tf/keras/metrics) or function.
        # @param model_type: classifier or reggresor. Define the scikit_learn wrapper's type.
        # @param epochs: Number of epochs.
        # @param learning_rate (Optional): Learning rate of optimizer. Default: 0.001
        # @param callbacks (Optional): Parameters of EarlyStopping (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping).
        #                        Default: None
        # @param validation_data (Optional): Validation data for cross validation. Default: None
        #                              tuple (x_val, y_val) of Numpy arrays or tensors
        #                              tuple (x_val, y_val, val_sample_weights) of Numpy arrays
        # @param optimizer (Optional): Optimizer's name. Supported 'adam' and 'rmsprop'. Default: adam
        # @param out_activation (Optional): Out layer's activation function. (see: https://www.tensorflow.org/api_docs/python/tf/keras/activations)
        #                                   Default: softmax
        # @param layer_activation (Optional): All layers' (except output layer) activation function.
        #                                   (see: https://www.tensorflow.org/api_docs/python/tf/keras/activations) Default: relu
        # @param batch_size (Optional): Size of batches. Default: None
        # @param return_probability (Optional): Return of predict function. If True predict_proba will be returned otherwise predicted classes
        # @param seed (Optional): Seed number of model. Defafult: 24
        
        self.layers=layers
        self.input_shape=input_shape
        self.loss=loss
        self.metrics=metrics
        self.model_type=model_type
        if optimizer == 'adam':
            self.optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            self.optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        self.out_activation=out_activation
        self.layer_activation=layer_activation
        self.seed=seed
        self.epochs = epochs
        self.batch_size=batch_size
        self.return_probability=return_probability
        self.learning_rate=learning_rate
        if callbacks == None:
            self.callbacks=None
        else:
            self.callbacks= tf.keras.callbacks.EarlyStopping(**calbacks)
        self.validation_data=validation_data
        self.initial_weights = None
        
        self.__create_model()
        self.__create_scikit()
        
        self.initial_weights=self.clf.get_weights().copy()
        
    def __create_model(self):
        tf.random.set_seed(self.seed)
        self.clf = tf.keras.Sequential()
        
        for i,el in enumerate(self.layers):
            if i == 0:
                self.clf.add(tf.keras.layers.Flatten(input_shape=self.input_shape))
                continue
            if i == len(self.layers)-1:
                self.clf.add(tf.keras.layers.Dense(el[0], activation=self.out_activation))
                continue
                
            reg= None
            if len(el) > 2:
                if el[2] == 'L1':
                    reg = tf.keras.regularizers.l1(l1=el[3])
                elif el[2] == 'L2':
                    reg = tf.keras.regularizers.l2(l2=el[3])
                elif el[2] == 'L1L2':
                    reg = tf.keras.regularizers.l1_l2(l1=el[4], l2=el[4])
            
            self.clf.add(tf.keras.layers.Dense(el[0], activation=self.layer_activation, kernel_regularizer = reg))
            if el[1] != 0:
                self.clf.add(tf.keras.layers.Dropout(rate=el[1]))             
                
        self.clf.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        
    def __get_model(self):
        return self.clf
    
    def __create_scikit(self):
        if self.model_type == 'classifier':
            self.sck =  tf.keras.wrappers.scikit_learn.KerasClassifier(self.__get_model, epochs = self.epochs, batch_size=self.batch_size, verbose=0)
        elif self.model_type == 'regressor':
            self.sck = tf.keras.wrappers.scikit_learn.KerasRegressor(self.__get_model, epochs = self.epochs, batch_size=self.batch_size, verbose=0)
            
    def summary(self):
        # Returns summary of network
        
        self.clf.summary()
    
    def fit(self, x, y, **kwargs):
        # Fits to training set
        # @param x: Training data
        # @param y: Training labels
        # @return history: A History object. Its History.history attribute is a record of training loss values and metrics values at
        #                  successive epochs, as well as validation loss values and validation metrics values (if applicable).
        #                  (https://www.tensorflow.org/api_docs/python/tf/keras/Model)
        self.history = self.sck.fit(x,y, callbacks=self.callbacks, validation_data=self.validation_data, **kwargs)
    
    def predict(self, x):
        # @return: Predicted classes or possibilities of classes of x
        # @param x: Test data.
        
        if self.return_probability:
            return self.sck.predict_proba(x)
        else:
            return self.sck.predict(x)
        
    def reset_weights(self):
        # Resets the learned weights.
        self.clf.set_weights(self.initial_weights)
    
    def score(self,x,y):
        # @return: Score of test data
        # @param x: Test data.
        # @param y: Test labels.
        return self.sck.score(x,y)

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor

class ANNWrapper(object):
    # This class is tensorflow based wrapper object that crated to work with scikit
    # Needs tensorflow to be imported. (import tensorflow as tf)
    # Tensorflow version: 2.4.0
    # Python version: 3.6.9
    # For more detailed information about tensorflow: https://www.tensorflow.org/api_docs/python/tf/all_symbols
    #                                                 https://www.tensorflow.org/api_docs/python/tf/keras/Model
    
    def __init__(self, model_type, layers, dropout, input_shape, loss, metrics, epochs, kernel_init='glorot_uniform',
                 l1_reg=None, l2_reg=None, learning_rate=0.001, callbacks=None, validation_data=None,
                 optimizer='adam', out_activation='softmax', layer_activation='relu', batch_size=None,
                 return_probability=False, seed=24):
        # @param layers: List of tuple that carries information about networks layers. Each tuple must covers at least 2 elements.
                # Ex. [(32, 0.2, 'L1', 0.01), (64, 0, 'L1L2', 0.01, 0.02) ]
                # First element represents Dense node count in layer
                # Second elemen represents dropout rate - float [0,1)
                # (Optional) Third element represents kernel regularizer type. L1, L2 or L1L2
                # (Not Optional if Regularizer Type given) Fourth (Fifth) element represents regularization factor.  
        # @param input_shape: Shape of input tensor.
        # @param loss: Loss function's name (see: https://www.tensorflow.org/api_docs/python/tf/keras/losses) or loss function.
        # @param metrics: List of metrics to be evaluated by the model during training and testing. 
        #                 Can be function's name (see: https://www.tensorflow.org/api_docs/python/tf/keras/metrics) or function.
        # @param model_type: classifier or reggresor. Define the scikit_learn wrapper's type.
        # @param epochs: Number of epochs.
        # @param learning_rate (Optional): Learning rate of optimizer. Default: 0.001
        # @param callbacks (Optional): Parameters of EarlyStopping (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping).
        #                        Default: None
        # @param validation_data (Optional): Validation data for cross validation. Default: None
        #                              tuple (x_val, y_val) of Numpy arrays or tensors
        #                              tuple (x_val, y_val, val_sample_weights) of Numpy arrays
        # @param optimizer (Optional): Optimizer's name. Supported 'adam' and 'rmsprop'. Default: adam
        # @param out_activation (Optional): Out layer's activation function. (see: https://www.tensorflow.org/api_docs/python/tf/keras/activations)
        #                                   Default: softmax
        # @param layer_activation (Optional): All layers' (except output layer) activation function.
        #                                   (see: https://www.tensorflow.org/api_docs/python/tf/keras/activations) Default: relu
        # @param batch_size (Optional): Size of batches. Default: None
        # @param return_probability (Optional): Return of predict function. If True predict_proba will be returned otherwise predicted classes
        # @param seed (Optional): Seed number of model. Defafult: 24
        
        self.layers=layers
        self.input_shape=input_shape
        self.loss=loss
        self.metrics=metrics
        self.model_type=model_type
        if optimizer == 'adam':
            self.optimizer=Adam(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            self.optimizer=RMSprop(learning_rate=learning_rate)
        self.out_activation=out_activation
        self.layer_activation=layer_activation
        self.kernel_init = kernel_init
        self.dropout = dropout
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.seed=seed
        self.epochs = epochs
        self.batch_size=batch_size
        self.return_probability=return_probability
        self.learning_rate=learning_rate
        if callbacks == None:
            self.callbacks=None
        else:
            self.callbacks= callbacks
        self.validation_data=validation_data
        self.initial_weights = None
        
        self.__create_model()
        self.__create_scikit()
        
        #self.initial_weights=self.clf.get_weights().copy()
        
    def __create_model(self):
        tf.random.set_seed(self.seed)
        self.clf = Sequential()
        
        for i,el in enumerate(self.layers):
            
            reg= None
            if self.l1_reg != None and self.l2_reg == None:
                reg = l1(l1=self.l1_reg)
            elif self.l1_reg == None and self.l2_reg != None:
                reg = l2(l2=self.l2_reg)
            elif self.l1_reg != None and self.l2_reg != None:
                reg = l1_l2(l1=self.l1_reg, l2=self.l2_reg)
                    
            if i == 0:
                self.clf.add(Dense(el,
                             input_dim=self.input_shape,
                             activation=self.layer_activation,
                             kernel_initializer = self.kernel_init,
                             kernel_regularizer = reg)
                            )

            elif i == len(self.layers)-1:
                self.clf.add(Dense(el,
                                   kernel_initializer = self.kernel_init,
                                   activation=self.out_activation)
                            )
            
            else:
                self.clf.add(Dense(el,
                                   activation=self.layer_activation,
                                   kernel_initializer = self.kernel_init, 
                                   kernel_regularizer = reg)
                            )
                
            if self.dropout != 0 and i != len(self.layers)-1:
                self.clf.add(Dropout(rate=self.dropout))             
                
        self.clf.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        
    def __get_model(self):
        return self.clf
    
    def __create_scikit(self):
        if self.model_type == 'classifier':
            self.sck =  KerasClassifier(self.__get_model, epochs = self.epochs, batch_size=self.batch_size, verbose=1)
        elif self.model_type == 'regressor':
            self.sck = KerasRegressor(self.__get_model, epochs = self.epochs, batch_size=self.batch_size, verbose=1)
            
    def summary(self):
        # Returns summary of network
        
        self.clf.summary()
    
    def fit(self, x, y, **kwargs):
        # Fits to training set
        # @param x: Training data
        # @param y: Training labels
        # @return history: A History object. Its History.history attribute is a record of training loss values and metrics values at
        #                  successive epochs, as well as validation loss values and validation metrics values (if applicable).
        #                  (https://www.tensorflow.org/api_docs/python/tf/keras/Model)
        self.history = self.sck.fit(x,y, callbacks=self.callbacks, validation_data=self.validation_data, **kwargs)
    
    def predict(self, x):
        # @return: Predicted classes or possibilities of classes of x
        # @param x: Test data.
        
        if self.return_probability:
            return self.sck.predict_proba(x)
        else:
            return self.sck.predict(x)
        
    def reset_weights(self):
        # Resets the learned weights.
        self.clf.set_weights(self.initial_weights)
    
    def score(self,x,y):
        # @return: Score of test data
        # @param x: Test data.
        # @param y: Test labels.
        return self.sck.score(x,y)