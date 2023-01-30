import pandas as pd
import numpy as np
import os
import unicodedata
import tensorflow as tf
import holidays_co

def strip_accents(text):
    """
    Strip accents from input String.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.

    reference: https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string
    hexaJer cooment on Jul 24 2015. Edited Nov 23 2017.
    """
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError): # unicode is a default on python 3
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text).lower()

def read_data(path):
    df = pd.read_parquet(path)

    #Clean dataset
    # df.columns = [strip_accents(col) for col in df.columns]
    # stations = df.columns[df.columns.str.contains("\(")]
    # df = df.groupby('timestamp')[stations].sum()
    # df = df[df.index <= pd.Timestamp('2021-04-30 23:45:00')]
    return df

def add_cycles(df, aggregation = 'day'):
    timestamp_s = df.index.map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    week = day * 7
    year = day * 365.2524

    df['year_sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['year_cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    df['week_sin'] = np.sin(timestamp_s * (2 * np.pi / week))
    df['week_cos'] = np.cos(timestamp_s * (2 * np.pi / week))

    if aggregation == 'day':
        return df
    else:
        df['day_sin'] = np.sin(timestamp_s * (2 * np.pi / day))
        df['day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))
        return df

def add_holidays(df):
    #Holidays Information
    years = range(2015,2022)
    holidays = []
    for year in years:
        year_holidays = holidays_co.get_colombia_holidays_by_year(year)
        for day in year_holidays:
            holidays.append(day.date)

    holidays = pd.Series(pd.to_datetime(holidays))

    # Holidays
    time_ = df.index.normalize()
    sundays = pd.Series((df.index.weekday == 6).astype(int))
    df_holidays = time_.isin(holidays)
    final_holidays = sundays.mask(df_holidays, 1)
    return final_holidays.values

def temporal_variables(df, aggregation = 'day'):
    add_cycles(df, aggregation)
    df['holiday'] = add_holidays(df)
    df['saturday'] = pd.Series((df.index.weekday == 5).astype(int)).values
    return df

def aggreagtion_func(df, aggregation = '15-mins'):
    """
    Aggregates transactions by the given aggregation parameter.

    Parameters:
    -----------
    - df: Pandas DataFrame,
        Transactions by station.
    - aggregation: str, default = '15-mins'.
        Aggregation interval {'15-mins','hour','day','month'}
    """

    if aggregation == '15-mins':
        hour = df.index.hour
        df = df[~hour.isin([0,1,2,3,23])]
        return df

    elif aggregation == 'hour':
        hours = df.resample('H').sum()
        hour = hours.index.hour
        return hours[~hour.isin([0,1,2,3,23])]

    elif aggregation == 'day':
        hour = df.index.hour
        df = df[~hour.isin([0,1,2,3,23])]
        return df.resample('D').sum()

    else:
        raise ValueError ('parameter {} not understood. Aggregation one of {15-mins,hour,day,month}'.format(aggregation))

def train_index(df, train_date):
    try:
        date = pd.Timestamp(train_date)
        idx = df.index.get_loc(date)

    except KeyError:
        date = pd.Timestamp(train_date) + pd.DateOffset(hours=4)
        idx= df.index.get_loc(date)

    return idx

def clean_data(path, aggregation = "15-mins"):
    df = read_data(path)

    to_drop_stations = ['(40000) cable portal tunal',
                        '(40001) juan pablo ii',
                        '(40002) manitas',
                        '(40003) mirador del paraiso']

    df = df.drop(columns=to_drop_stations)
    df = aggreagtion_func(df, aggregation = aggregation)
    df = temporal_variables(df, aggregation = aggregation)
    return df

def min_max(series, desired_range = None):
    max_value = series.max()
    min_value = series.min()
    min_max_norm =  (series - min_value)/(max_value - min_value)

    if desired_range:
        t_max = desired_range[0]
        t_min = desired_range[1]
        min_max_norm = min_max_norm * (t_max - t_min) + t_min
    return min_max_norm


def read_stations(path):
    stations_df = pd.read_parquet(path)
    stations_df['station_name'] = [strip_accents(col) for col in stations_df.station_name.astype(str)]
    stations_df = stations_df[['station_name','latitude','longitude']].drop_duplicates(subset = 'station_name')
    stations_df['latitude'] = min_max(stations_df['latitude'], (-5,5))
    stations_df['longitude'] = min_max(stations_df['longitude'], (-5,5))

    return stations_df


def tf_data(transactions_path, stations_path, aggregation, train_date, max_transactions=None, max_stations=None):

    df = clean_data(transactions_path, aggregation =  aggregation)
    stations_df = read_stations(stations_path)
    exog_vars = list(set(df.columns[~df.columns.str.contains("\(")]))
    exog_vars = [
                'year_sin', 'year_cos',
                 'week_sin', 'week_cos',
                 'day_sin', 'day_cos',
                 'holiday', 'saturday'
                ]
    list_stations = list(df.columns[df.columns.str.contains("\(")])

    if max_stations:
        list_stations = list_stations[:int(max_stations)]
    if max_transactions:
        df = df.iloc[:int(max_transactions)]

    # print("DF Shape: {}".format(df.shape))

    stations_df = stations_df[stations_df.station_name.isin(list_stations)].set_index('station_name').T
    stations_df = stations_df[list_stations] #Same order as df

    window_length = 1065
    idx_features = [-1065, -533, -153, -77] + list(range(-41,-33)) + list(range(-7,-1))
    idx_PE = [-1065, -533, -153, -77] + list(range(-41,-33)) + list(range(-7,0))
    train_idx = train_index(df, train_date) - window_length - 1

    labels_list = []
    features_list = []
    PE_list = []
    for i in range(len(df) - window_length + 1):
        x = slice(i ,i + window_length)
        window = df[x]
        labels = window[list_stations][-1:]
        features = window[list_stations].iloc[idx_features]
        possition = window[exog_vars].iloc[idx_PE]

        labels_list.append(np.array(labels))
        features_list.append(np.array(features))
        PE_list.append(possition)

    labels = tf.squeeze(tf.convert_to_tensor(np.array(labels_list)))
    features = tf.convert_to_tensor(np.array(features_list))
    time_embeddings = tf.convert_to_tensor(np.array(PE_list))

    # Convert Spatial embeddings in a 3D tensor
    spatial_embeddings = tf.convert_to_tensor(stations_df.values.T)
    spatial_embeddings = spatial_embeddings[tf.newaxis,:,:]
    spatial_embeddings = tf.tile(spatial_embeddings, tf.convert_to_tensor([features.shape[0],1,1]))

    # Open/Close status as a dummy variable
    status_f = tf.where(features == 0, x = features * 0, y = 1)
    status_l = tf.where(labels == 0, x = labels * 0, y = 1)
    status_l = tf.reshape(status_l, (-1, 1, 147))

    status = tf.concat([status_f, status_l], axis = 1)
    

    # Split train and test
    data_points = [features, time_embeddings, spatial_embeddings, labels, status]
    names = ['features', 'time_embeddings', 'spatial_embeddings', 'labels', 'status']
    train_data = {}
    test_data = {}
    print('')
    for name, element in zip(names,data_points):
        train_data[name] = element[:train_idx]
        test_data[name] = element[train_idx:]
        print('Train {} shape: {}'.format(name, train_data[name].shape))
        print('Test {} shape: {}'.format(name, test_data[name].shape))
        print('')

    metadata = {'list_stations':list_stations,
                'train_date_index':df.index[window_length - 1:train_idx + window_length -1],
                'test_date_index':df.index[train_idx + window_length -1:]}

    # print(metadata['train_date_index'].shape)
    # print(metadata['test_date_index'].shape)

    return train_data, test_data, metadata
