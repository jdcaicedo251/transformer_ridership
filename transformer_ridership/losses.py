import pandas as pd
import numpy as np


def _validate_target_predicted(target, predicted):
    """
    Validates target and predicted data. Both arguments 
    must have the same shape. If argument is a DataFrame, 
    this function converts it to a numpy array. 
    """
    
    if isinstance(target, pd.DataFrame) or isinstance(target, pd.Series):
        target = target.values
        
    if isinstance(predicted, pd.DataFrame) or isinstance(predicted, pd.Series):
        predicted = predicted.values
    
    assert isinstance(target, np.ndarray)
    assert isinstance(predicted, np.ndarray)
    assert target.shape == predicted.shape
    
    return target, predicted

def stats(target, predicted, func, bs_se = False):
    if bs_se:
        return func(target, predicted), bootstrap_se(target, predicted, func)
    return func(target, predicted)

def MAAPE(target, predicted, axis = None):
    """ Mean Abosulute Arctangent Percentual Error """
    target, predicted = _validate_target_predicted(target, predicted)
        
    return 100 * np.arctan(np.abs((target - predicted)/(target + 1e-9))).mean(axis = axis)

def wMAPE(target, predicted, axis = None):
    """ Weighted Mean Abosulute Percentual Error """
    target, predicted = _validate_target_predicted(target, predicted)
    
    num = np.abs(target - predicted).sum(axis = axis)
    dem = np.abs(target).sum(axis = axis)
    return 100*num/dem

def sMAPE(target, predicted, axis = None):
    """ symetric Mean Abosulute Percentual Error """
    target, predicted = _validate_target_predicted(target, predicted)
    
    num = np.abs(target-predicted)
    dem = np.abs(target) + np.abs(predicted)
    return 100*np.nan_to_num(num/dem).mean(axis = axis)

def RMSE(target, predicted, axis = None):
    """ symetric Mean Abosulute Percentual Error """
    target, predicted = _validate_target_predicted(target, predicted)
   
    mse = np.mean((target-predicted)**2, axis = axis)
    return np.sqrt(mse)

def MAE(target, predicted, axis = None):
    """ symetric Mean Abosulute Percentual Error """
    target, predicted = _validate_target_predicted(target, predicted)
    return np.mean(np.abs(target-predicted), axis = axis)

def bootstrap_se(target, predicted, func):
    """ Bootsrap function to estimate the standard error of 'func'
    
    Parameters: 
    - target: 1-d array-like. Target data of size n
    - predicted: 1-d array-like. Predicted data of size n
    - func: func. inputs of the function are target, predicted and axis. 
    
    Return: 
    estimated standard error of 'func' via bootstrap. 
    """
    target, predicted = _validate_target_predicted(target, predicted)
    target = target.flatten()
    predicted = predicted.flatten()
    
    indices = np.random.choice(target.size, size = (1000,1000), replace=True)
    target_sample = target[indices]
    predicted_sample = predicted[indices]
    
    return func(target_sample, predicted_sample, axis = 1).std()

def summary_erros(target, predictions, filter_obs = None):
    """
    Summary table for accuracy metrics
    
    Parameters:
    - target: 
    - predictions: list. List of dataframes. Each dataframe 
                   must contain df.name where name is the the 
                   model where the prediction is from. 
    """
    
    assert isinstance(predictions, list)
    error_list = [RMSE, MAE, MAAPE, wMAPE, sMAPE]
    error_name = ['RMSE', 'MAE','MAAPE', 'wMAPE', 'sMAPE']
    results = {}
    
    if filter_obs:
        target = target[:]
    
    
    for p in predictions:
        model = p.name
        error_dict = {}
        
        if filter_obs:
            t = target[filter_obs[0]:filter_obs[1]]
            p = p[filter_obs[0]:filter_obs[1]]
        else:
            t = target
        
        for func, name in zip(error_list, error_name):
            error = func(t, p)
            std = bootstrap_se(t, p, func)
            error_dict[name] = "{:.2f} (\u00B1{:.2f})".format(error,1.96*std)
        
        results[model] = error_dict
        
    print(pd.DataFrame(results).T.style.to_latex(caption = 'Summary accuracy metrics', 
                                                 position_float='centering', 
                                                 hrules = True))
