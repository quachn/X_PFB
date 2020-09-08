
from dateutil import parser
import pandas as pd
import numpy as np
from scipy import stats
import datetime
import math
from math import cos, sin, atan2, sqrt, radians, degrees
import json
from helpers.gps_coordinates import GpsUtils
from vincenty import vincenty
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder

def missing_values_table(df):
    # Nb of Nan per column and %
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum()/len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table = mis_val_table[mis_val_table['Missing Values'] > 0].sort_values(by=['% of Total Values'], ascending=[0])
    return mis_val_table

# -----
def add_time_features(df):
    # # Time features: wwejday, month, day, hour
    # selection time - decomposition
    # Monday=0, Sunday=6
    df['selection time'] = pd.to_datetime(df['selection time'], infer_datetime_format=True)
    df['selection_weekday'] = df['selection time'].dt.dayofweek
    df['selection_month'] = df['selection time'].dt.month
    df['selection_day'] = df['selection time'].dt.day
    df['selection_hour'] = df['selection time'].dt.hour

    # Holidays
    years = df['selection time'].dt.year.unique()
    import holidays
    holidays = holidays.France(years=years)
    df['selection_is_holiday'] = df.apply(axis=1, func=lambda x: x['selection time'] in holidays)
    return df

# Spatial features
def add_spatial_features(df):
    # No need to calculate distanve, use OSRM estimated distance
    # from vincenty import vincenty
    # df['intervention_distance'] = df.apply(lambda x: vincenty((x['latitude before departure'], x['longitude before departure']),
    #                                                          (x['latitude intervention'], x['longitude intervention']))*1000,
    #                                       axis=1
    #                                      )
    df['OSRM_estimated_speed'] = df['OSRM estimated distance'] / df['OSRM estimated duration'] # speed in m/s
    df['departure2intervention_bearing'] = df.apply(lambda x: calculate_initial_compass_bearing(
        (x['latitude before departure'], x['longitude before departure']),
        (x['latitude intervention'], x['longitude intervention'])),
        axis=1)
    df['mid_point'] = df.apply(lambda x: mid_point(
        x['latitude before departure'], x['longitude before departure'],
        x['latitude intervention'], x['longitude intervention']),
        axis=1)
    df[['mid_point_lat', 'mid_point_lon']] = pd.DataFrame(df['mid_point'].values.tolist(), index=df.index)
    df.drop('mid_point', axis=1, inplace=True)

def add_distances_from_paris_center(df):
    # -- 20200411 - add distance & bearing from Paris-center to intervention place
    for place, lat_col, lon_col in [
        ('intervention', 'latitude intervention', 'longitude intervention'),
        ('departure', 'latitude before departure', 'longitude before departure'),
        ('mid_point', 'mid_point_lat', 'mid_point_lon'),
        ('waypoint1', 'waypoint1_lat', 'waypoint1_lon'),
        ('waypoint2', 'waypoint2_lat', 'waypoint2_lon'),
        ]:
        df[f'paris2{place}_bearing'] = df.apply(lambda x: calculate_initial_compass_bearing(
            (GpsUtils.paris_center_lat, GpsUtils.paris_center_lon),
            (x[lat_col], x[lon_col])
            ),
            axis=1)
        df[f'paris2{place}_km'] = df.apply(lambda x: vincenty(
            (GpsUtils.paris_center_lat, GpsUtils.paris_center_lon),
            (x[lat_col], x[lon_col])
            ),
            axis=1)


def calculate_initial_compass_bearing(pointA, pointB):
    """
    https://gist.github.com/jeromer/2005586
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
                                           * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

# Drop features that are not explotable
# def drop_features(df):
#     df.drop([
#         'emergency vehicle selection',  # ID number
#         'intervention',  # ID number, strongly correlated to the previous
#         'selection time',  # Date Time, already splitted
#         'delta position gps previous departure-departure',  # 97% missing
#         'GPS tracks departure-presentation',  # 70% missing
#         'GPS tracks datetime departure-presentation',  # 70% missing
#         'OSRM response',
#
#     ], axis=1, inplace=True)
#     return df
#
#     x = math.sin(diffLong) * math.cos(lat2)
#     y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
#                                            * math.cos(lat2) * math.cos(diffLong))
#
#     initial_bearing = math.atan2(x, y)
#
#     # Now we have the initial bearing but math.atan2 return values
#     # from -180° to + 180° which is not what we want for a compass bearing
#     # The solution is to normalize the initial bearing as shown below
#     initial_bearing = math.degrees(initial_bearing)
#     compass_bearing = (initial_bearing + 360) % 360
#
#     return compass_bearing


def mid_point(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    Bx = cos(lat2) * cos(dlon)
    By = cos(lat2) * sin(dlon)
    lat_m = atan2(sin(lat1) + sin(lat2), sqrt((cos(lat1) + Bx)**2 + By**2))
    lon_m = lon1 + atan2(By, cos(lat1) + Bx)
    return list(map(degrees,[lat_m, lon_m]))

def extract_waypoints(js_dict, i, field):
    if js_dict.get('waypoints', None):
        try:
            if field == 'name':
                return js_dict['waypoints'][i][field]
            if field == 'lon':
                return js_dict['waypoints'][i]['location'][0]
            if field == 'lat':
                return js_dict['waypoints'][i]['location'][1]
        except KeyError as e:
            print(e, 'in extract_waypoints')
            print(js_dict)
            #raise e
    return None

def add_waypoints(df):
    js_col = df['OSRM response'].apply(json.loads)
    for i in range(1,3):
        df[f'waypoint{i}_name'] = js_col.apply(lambda x: x['waypoints'][i-1]['name'])
        df[f'waypoint{i}_lon'] = js_col.apply(lambda x: x['waypoints'][i-1]['location'][0])
        df[f'waypoint{i}_lat'] = js_col.apply(lambda x: x['waypoints'][i-1]['location'][1])
        df[f'waypoint{i}_name'].fillna('Missing', inplace=True)
        df[f'waypoint{i}_name'] = df[f'waypoint{i}_name'].replace('', 'Missing')
    # additional file waypoints
    js_col = df['OSRM estimate from last observed GPS position'].dropna().apply(json.loads)
    for i in range(1,3):
        df[f'waypoint{i+2}_name'] = js_col.apply(lambda x: extract_waypoints(x, i-1, 'name'))
        df[f'waypoint{i+2}_lon'] = js_col.apply(lambda x: extract_waypoints(x, i - 1, 'lon'))
        df[f'waypoint{i + 2}_lat'] = js_col.apply(lambda x: extract_waypoints(x, i - 1, 'lat'))
        # Missing
        df[f'waypoint{i+2}_name'].fillna('Missing', inplace=True)
        df[f'waypoint{i+2}_name'] = df[f'waypoint{i}_name'].replace('', 'Missing')
        df[f'waypoint{i+2}_lat'] = df[f'waypoint{i+2}_lat'].fillna(0)
        df[f'waypoint{i+2}_lon'] = df[f'waypoint{i+2}_lon'].fillna(0)

def parse_emergency_vehicle_type(x):
    # Decomopose 'emergency vehicle type' into 'vehicule_type' and 'vehicule_ownrer'
    l = x['emergency vehicle type'].split(' ')
    if len(l) > 1:
        return (l[0], l[1])
    return (x['emergency vehicle type'], 'Missing')

def add_vehicule_type_info(df):
    # Decomopose 'emergency vehicle type' into 'vehicule_type' and 'vehicule_ownrer'
    df[['vehicule_type', 'vehicule_ownrer']] = \
        pd.DataFrame(df.apply(parse_emergency_vehicle_type, axis=1).tolist(),
                     index=df.index)

def replace_gps_coord(df):
    """
    Replaces GPS coordinates by cartesian coordinates.
    The cartesian coords are based on a local plan on Paris center
    :param df:
    :return:
    """
    for old_lat, old_lon, new_x, new_y in [
        ('latitude before departure', 'longitude before departure', 'departure_xEst', 'departure_yNorth'),
        ('latitude intervention', 'longitude intervention', 'intervention_xEst', 'intervention_yNorth'),
        ('mid_point_lat', 'mid_point_lon', 'mid_point_xEst', 'mid_point_yNorth'),
        ('waypoint1_lat', 'waypoint1_lon', 'waypoint1_xEst', 'waypoint1_yNorth'),
        ('waypoint2_lat', 'waypoint2_lon', 'waypoint2_xEst', 'waypoint2_yNorth'),
    ]:
        df['tmp_coords'] = df.apply(lambda x: GpsUtils.GeodeticToEnu(x[old_lat], x[old_lon]), axis=1)
        df[[new_x, new_y, 'tmp_zUp']] = pd.DataFrame(df['tmp_coords'].values.tolist(), index=df.index)
        df.drop(['tmp_coords', 'tmp_zUp',
                 old_lat,old_lon,
                 ], axis=1, inplace=True)

def impute_missing_by_mode(df, x_train):
    df['location of the event'] = df['location of the event'].fillna(
        x_train['location of the event'].mode()[0])
    df['updated OSRM estimated duration'] = df['updated OSRM estimated duration'].fillna(0)
    df['OSRM estimated duration from last observed GPS position'] = df['OSRM estimated duration from last observed GPS position'].fillna(0)
    df['OSRM estimated distance from last observed GPS position'] = df['OSRM estimated distance from last observed GPS position'].fillna(0)
    df['time elapsed between selection and last observed GPS position'] = df['time elapsed between selection and last observed GPS position'].fillna(0)
    df['delta position gps previous departure-departure'] = df['delta position gps previous departure-departure'].fillna(0)
    for c in ['GPS_Tracks_80pc_kmh', 'GPS_Tracks_std_kmh', 'GPS_Tracks_mean_kmh']:
        df[c] = df[c].fillna(0)

def drop_features(df):
    df.drop([
        'emergency vehicle selection',  # ID number
        'intervention',  # ID number, strongly correlated to the previous
        'selection time',  # Date Time, already splitted
        #'delta position gps previous departure-departure',  # 97% missing
        'GPS tracks departure-presentation',  # parsed
        'GPS tracks datetime departure-presentation',  # parsed
        'OSRM response', # parsed
        'OSRM estimate from last observed GPS position',
        'emergency vehicle type', # decomposed to 'vehicule_type' and 'vehicule_owner'
    ], axis=1, inplace=True)
    return df

def onehot(df):
    vars_to_encode = [
        'alert reason category',
        'alert reason',
        # 'intervention on public roads', # boolean int
        'floor',
        'location of the event',
        # 'emergency vehicle',
        # 'emergency vehicle type',  # string
        'vehicule_type', 'vehicule_ownrer',
        'rescue center',
        'status preceding selection',  # string, only 2 values
        # 'departed from its rescue center', # boolean int
        # 'selection_weekday',
        # 'selection_month',
        # 'selection_day',
        # 'selection_hour',
        # 'selection_is_holiday',  # boolean int
        # Removed because it will overuse memory
        # 'waypoint1_name',
        # 'waypoint2_name',
        'intervention_place',
    ]

    df1hot = pd.get_dummies(
        data=df,
        columns=vars_to_encode,
        prefix=['H_{}'.format(c) for c in vars_to_encode],  # prefix column names for dummies
        drop_first=True,  # keep only (k-1) levels
        sparse=False,  # True -> save encoded cols to a sparseArray
    )
    # Drops columns
    df1hot.drop([
        'waypoint1_name',
        'waypoint2_name',
        'waypoint3_name',
        'waypoint4_name',
    ],
        axis=1,
        inplace=True)
    return df1hot

def convert_types(df):
    df['location of the event'] = df['location of the event'].astype('int')

def add_mean_speeds(x_train, y_train, xs_to_apply):
    '''
    Calculates the mean speed for every coordinate (rounded by 2 decimals).
    :param x_train: dataframe of train data
    :param y_train: dataframe of label data
    :param x_to_apply: list of dataframes to be applied using the means computed from train data
    :return:
    '''
    # Learns the avg speed on train data
    # aggregate by latitudex-longitude
    lat_colname = 'latitude intervention'
    lon_colname = 'longitude intervention'
    val_colname = 'speed_mean_kmh'
    # Concatenate x and y
    df = pd.concat([x_train, y_train], axis=1)
    df[val_colname] = df['OSRM estimated distance'] * 3.6 / df['delta departure-presentation']
    z = df[[lat_colname, lon_colname, val_colname]]
    # round lat/lon
    z[lat_colname] = z[lat_colname].round(2)
    z[lon_colname] = z[lon_colname].round(2)
    z = z.groupby([lat_colname, lon_colname]).agg('mean')
    z.reset_index(inplace=True)
    # creates crosstable: a 2D matrix: means[lat, lon]
    z = z.pivot(index=lat_colname, columns=lon_colname, values=val_colname)
    # apply
    for x_apply in xs_to_apply:
        x_apply[val_colname] = x_apply.apply(
            lambda x: z.loc[
                np.round(x[lat_colname],2),
                np.round(x[lon_colname], 2),
                ],
            axis=1,
        )
        x_apply[val_colname].fillna(45.0, inplace=True) # probably situated far from paris


def add_intervention_place(x_train, x_test, n_bins=20):
    """
    Add the intervention place as a single categorical variable.
    Discretize the latitide & longitude intervention by n_bins.
    Then, number the grid cell from 0 to 20x20.
    :param x_train:
    :param x_test:
    :return: None
    New column 'intervention_place' is added into x_train and x_test.
    """
    var_encoders = {}

    for var in ['latitude intervention', 'longitude intervention']:
        new_var = f'bin_{var}'
        var_encoders[var] = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
        x_train[new_var] = var_encoders[var].fit_transform(x_train[var].values[:, np.newaxis])
        x_test[new_var] = var_encoders[var].transform(x_test[var].values[:, np.newaxis])

    x_train['intervention_place'] = x_train.apply(lambda x:
                                                  x['bin_latitude intervention'] * n_bins + x['bin_longitude intervention'],
                                                  axis=1)
    x_test['intervention_place'] = x_test.apply(lambda x:
                                                x['bin_latitude intervention'] * n_bins + x['bin_longitude intervention'],
                                                axis=1)
    x_train.drop(['bin_latitude intervention', 'bin_longitude intervention'], axis=1, inplace=True)
    x_test.drop(['bin_latitude intervention', 'bin_longitude intervention'], axis=1, inplace=True)
    x_train.intervention_place = x_train.intervention_place.astype(int)
    x_test.intervention_place = x_test.intervention_place.astype(int)

# --- Extracts info from GPS Tracking (list of recorded GPS and their timestamps)

def tracking_gps_2_times_lonlat(x):
    '''
    Parses the columns 'GPS tracks datetime departure-presentation' and 'GPS tracks departure-presentation'
    :param x: a row of the dataframe
    :return: list of times, list of (longitude, latititude)
    '''
    # list of time
    times = x['GPS tracks datetime departure-presentation'].split(';')
    times = [parser.parse(s) for s in times]
    # list of (lon, lat)
    coords = x['GPS tracks departure-presentation'].split(';') # list of str 'lon;lat'
    lon_lat = [i.split(',') for i in coords]  # couples of (lon, lat)
    lon_lat = [(float(lon), float(lat)) for lon, lat in lon_lat]
    return times, lon_lat

def tracking_gps_2_durations_distances(x):
    times, lon_lat = tracking_gps_2_times_lonlat(x)
    durations = []  # durations in second
    distances = []  # distance in km
    t_prev, lon_prev, lat_prev = None, None, None
    for t, (lon, lat) in zip (times, lon_lat):
        if t_prev:
          durations.append((t - t_prev).total_seconds())
          distances.append(vincenty((lat_prev, lon_prev), (lat, lon)))
        t_prev, lon_prev, lat_prev = t, lon, lat
    return durations, distances

def tracking_gps_parse_info(x):
    '''
    Parses the GPS Tracking columns
    :return: a dataframe with
    - GPS_Tracks_records: number of records
    - GPS_Tracks_duration_hr: total duration in hour
    - GPS_Tracks_distance_km: total distance in km
    - GPS_Tracks_mean_kmh: mean speed in kmh
    - GPS_Tracks_std_kmh: std dev speed in kmh
    - GPS_Tracks_80pc_kmh: 80% percentile speed in kmh
    '''
    chck_field = x['GPS tracks datetime departure-presentation']
    if isinstance(chck_field, str) and len(chck_field) > 0:
        durations, distances = tracking_gps_2_durations_distances(x)
        tot_duration_hour = np.sum(durations) / 3600.0
        tot_distance_km = np.sum(distances)
        # avg_speed_kmh = tot_distance_km / tot_duration_hour
        # pointwise speeds
        speeds_kmh = [(dist * 3600.0 / dur)
                      for dur, dist in zip(durations, distances)
                      if dur > 1.0]
        try:
            pctile = np.percentile(speeds_kmh, 80)
        except:
            pctile = np.median(speeds_kmh)
        return len(durations) + 1, \
               tot_duration_hour, tot_distance_km, \
               np.mean(speeds_kmh), np.std(speeds_kmh), pctile
    else:
        # If tracking data is missing
        return (0,) * 6

def add_gps_tracking_info(df):
    # Extracts the info from the variable 'GPS Tracks ...'
    new_cols = [
        'GPS_Tracks_records',
        'GPS_Tracks_duration_hr',
        'GPS_Tracks_distance_km',
        'GPS_Tracks_mean_kmh',
        'GPS_Tracks_std_kmh',
        'GPS_Tracks_80pc_kmh',
    ]
    df[new_cols] = pd.DataFrame(df.apply(tracking_gps_parse_info, axis=1).tolist(),
                                index=df.index)

# ---

def load_all(data_dir):
    x_train = pd.read_csv(data_dir + 'x_train.csv')
    y_train = pd.read_csv(data_dir + 'y_train.csv')
    x_test = pd.read_csv(data_dir + 'x_test.csv')
    # additional files
    x_train_ad = pd.read_csv(data_dir + 'x_train_additional_file.csv')
    x_test_ad = pd.read_csv(data_dir + 'x_test_additional_file.csv')
    return x_train, y_train, x_test, x_train_ad, x_test_ad

def data_prep_all(data_dir ='data/'):
    x_train, y_train, x_test, x_train_ad, x_test_ad = load_all(data_dir)
    x_train = pd.concat([x_train, x_train_ad], axis=1)
    x_test = pd.concat([x_test, x_test_ad], axis=1)
    for df in [x_train, x_test]:
        add_time_features(df)
        add_spatial_features(df)
        add_waypoints(df)
        add_distances_from_paris_center(df)
        #replace_gps_coord(df)
        add_gps_tracking_info(df)
        add_vehicule_type_info(df)
        drop_features(df)
        impute_missing_by_mode(df, x_train)
        convert_types(df)
    add_mean_speeds(x_train, y_train, [x_train, x_test])
    add_intervention_place(x_train=x_train, x_test=x_test, n_bins=20)
    x_train_1hot = onehot(x_train)
    x_test_1hot = onehot(x_test)
    return x_train, y_train, x_test, x_train_1hot, x_test_1hot

def save_to_disk(dfs, file_names, DATA_DIR = 'data/'):
    for df, file_name in zip(dfs, file_names):
        df.to_csv(DATA_DIR + file_name, compression='zip', index=False)

def zzz():
    x_train, y_train, x_test, x_train_ad, x_test_ad = load_all(data_dir='data/')
    x_train = pd.concat([x_train, x_train_ad], axis=1)
    x_test = pd.concat([x_test, x_test_ad], axis=1)
    #add_waypoints(x_train)
    add_waypoints(x_test)
    return x_train, x_test

if __name__ == '__main__':
    print('Start...')
    # zzz()
    x_train, y_train, x_test, x_train_1hot, x_test_1hot = data_prep_all()
    # # saves to disk
    dfs = [x_train, x_test, x_train_1hot, x_test_1hot]
    file_names = ['x_train', 'x_test', 'x_train_1hot', 'x_test_1hot']
    file_names = [f + '.csv.zip' for f in file_names]
    save_to_disk(
        dfs=dfs,
        file_names=file_names,
        DATA_DIR='data/prepared/'
    )
