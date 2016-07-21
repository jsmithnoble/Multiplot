import pandas as pd
import numpy as np
import requests
import json
import folium
from time import time
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import cPickle as pickle
import os
import geopy
from sklearn.cluster import KMeans, DBSCAN


def get_data():
    '''
    return event merged with event_readings
    '''
    event_cols = ['event_id', 'weird_hash_thing', 'two', 'device_id', 'zero', 'NaNs', 'file_path?',
    '13-digit_int', 'lat', 'lon', 'device_type', 'two_again', 'device_type_again','device_id_again', 'zero_again', 'url_session_num', '10-digit-int','int_2_to_1101']
    event = pd.read_csv("event.csv", names = event_cols)
    event.set_index('event_id', inplace =True)
    event['timestamp'] = pd.to_datetime(event['10-digit-int'] - 4 * 60 * 60, unit = 's')#, format = "%Y-%m-%d")
    event = event.drop(['two_again', 'device_type_again', 'device_id_again', 'zero_again', '10-digit-int','url_session_num','int_2_to_1101','file_path?','NaNs','zero','weird_hash_thing','two','13-digit_int'], axis=1)
    readings_cols = ['event_id','value','measurment']
    readings = pd.read_csv("event_readings.csv", names = readings_cols)
    readings.set_index('event_id', inplace = True)
    readings = readings.pivot(columns = 'measurment', values = 'value')
    return event.join(readings).sort_values('timestamp')

def get_single_device(device_id):
    '''
    return a subset of data where device_id == device_id
    '''
    data = event[event['device_id'] == device_id].sort_values('timestamp')
    time_delta = [pd.tslib.Timedelta(1e3, unit = 'm')]
    for idx, time in enumerate(data.timestamp):
        if idx == 0:
            pass
        else:
            time_delta.append(time - data.timestamp.iloc[idx - 1])
    data['time_since_last_pt'] = time_delta
    ride_break = pd.tslib.Timedelta(15, unit = 'm')
    ride_num = 0
    ride = []
    for delta in data['time_since_last_pt']:
        if delta < ride_break:
            ride.append(ride_num)
        else:
            ride_num += 1
            ride.append(ride_num)
    data['ride'] = ride
    return data

def plot_single_device(device_id):
    '''
    INPUT device_id: string
    OUTPUT: None
    save plot of a single device_id
    '''
    data = get_single_device(device_id)
    rides = data.ride.unique()
    lat = data.lat.mean()
    lon = data.lon.mean()
    m = folium.Map((lat,lon), zoom_start = 12)
    for ride in rides:
        temp = data[data.ride == ride]
        coords = zip(temp.lat,temp.lon)
        time = temp.timestamp
        plot_single_ride(m, coords, time)
    m.save("plots/devices/{}".format(device_id))

def plot_single_ride(m, coords, time = None, interval = 5):
    '''
    INPUT:  m = folium Map object to be plotted on
            coords = list of tuples containing (latitude, longitude)
            time  = pandas series object containing timestamps which correspond to the coords
            interval = interval in minutes to place markers must be integer >= 1
    OUTPUT: folium Map object with polyline of the ride and markers if time is specified
    '''
    folium.PolyLine(coords).add_to(m)

    if not isinstance(time,type(None)):
        for idx, t in enumerate(time.astype(str)):
            if idx % interval == 0:
                folium.Marker(coords[idx], popup = t).add_to(m)
    folium.Marker(coords[0], popup = "Start").add_to(m)
    folium.Marker(coords[-1], popup = "End").add_to(m)
    return m

def find_major_cities(df):
    db = DBSCAN(eps = .1, min_samples = 100) # eps 1 = 111.2KM, 0.1 = 11.12KM
    labels = db.fit_predict(df[['lat','lon']])
    labels = np.expand_dims(labels, axis = 1)
    df['county'] = labels
    geocoder = geopy.geocoders.Nominatim()
    counties = {}
    centroids = {}
    exceptions = []
    for label in np.unique(labels):
        X = df[df.county == label][['lat','lon']]
        lat, lon = X.mean()
        location = geocoder.reverse((lat,lon))
        try:    # Works for most countries
            counties[label] = location.raw['address']['county']
            centroids[label] = (lat,lon)
        except:
            try:    # UK and CA use suburb instead of county
                counties[label] = location.raw['address']['suburb']
                centroids[label] = (lat,lon)
            except:     # Sweeden and Romania are in unicode so this is done by hand
                if (lat,lon) == (37.465285558094635, 126.95470294348861):
                    counties[label] = 'Sweeden'
                    centroids[label] = (lat,lon)
                elif (lat,lon) == (44.435190075805657, 26.093614125129637):
                    counties['label'] = 'Romania'
                    centroids[label] = (lat,lon)
                else:   # One set has weid behavior 923 events over 52 devices
                    exceptions.append((lat,lon))
                    counties[label] = (lat,lon)
                    centroids[label] = (lat,lon)
    df['city_center'] = df.county.map(centroids)
    df.county = df.county.map(counties)
    return exceptions



def find_city_centroids(df, n_clusters = 8, pollutant = 'PM25'):
    '''
    INPUT:  df = a dataframe with lat, lon, timestamp and [pollutant] columns
            n_clusters = the number of clusters to find
            pollutant = the type of pollutant to filter by
    OUTPUT: None

    This function appends a column to df with the lat lon of the nearest centroid
    '''
    mask = np.ma.masked_invalid(df[pollutant])
    X = df[~mask.mask][['lat', 'lon', 'timestamp']]
    start = X.timestamp.min()
    X['seconds'] = X.timestamp - start
    X['seconds'] = X.seconds.apply(lambda x: x.total_seconds())
    ts = X.pop('timestamp')
    X = (X - X.mean())/X.std()

    estimator = KMeans(n_clusters = n_clusters)
    labels = estimator.fit_predict(X)
    # labels = np.expand_dims(labels, axis = 1)
    X['centroid'] = labels
    df = df.join(X.centroid)
    centroids = {}
    for label in np.unique(labels):
        temp = df[df.centroid == label][['lat', 'lon']]
        lat, lon = temp.mean()
        centroids[label] = (lat, lon)
    df['centroid'] = df['centroid'].map(centroids)

# terminal command to connect to db and download event table and convert to csv
# psql -h 104.196.143.165 -U multitude -d d2c_api_prod -p 5432 -o event.csv -c 'select * from event;' -P format=unaligned -P tuples_only -P fieldsep=\,
# psql -h 104.196.143.165 -U multitude -d d2c_api_prod -p 5432 -o event_meta_data.csv -c 'select * from event_meta_data;' -P format=unaligned -P tuples_only -P fieldsep=\,
# psql -h 104.196.143.165 -U multitude -d d2c_api_prod -p 5432 -o event_readings.csv -c 'select * from event_readings;' -P format=unaligned -P tuples_only -P fieldsep=\,
# psql -h <host_ip> -U <username> -d <database> -p <port> -o <output_file_path> -c '<query>' -P format=unaligned -P tuples_only -P fieldsep=\,

if __name__ == '__main__':
    # event = get_data()
    # exceptions = find_major_cities(event)
    # event.to_pickle('event.pkl')
    event = pickle.load(open('event.pkl','rb'))

    # d1 = get_single_device('SAMSUNG-SM-G920A:1083')
    ny = event[event['county'] == "New York County"]
    find_city_centroids(ny)
    # lat = d1.lat.mean()
    # lon = d1.lon.mean()
    # days = d1.timestamp.dt.date.unique()
    # for n, day in enumerate(days):
    #     m = folium.Map((lat,lon), zoom_start = 13)
    #     temp = d1[d1.timestamp.dt.date == day]
    #     coords = zip(temp.lat,temp.lon)
    #     folium.PolyLine(coords).add_to(m)
    #     folium.Marker(coords[0], popup = 'Start').add_to(m)
    #     folium.Marker(coords[-1], popup = "End").add_to(m)
    # m.save('plots/devices/{}.html'.format(device_id))
    """Same user for 'Nexus-6P:1008' and 'AirBeam:001896105F77'
        ?Same user for 'Terrier-0006667BB110' and 'SAMSUNG-SM-G920A/1083'"""

    # plot_single_device('Nexus-6P:1008')
    # plot_single_device('AirBeam:001896105F77')
    # d2 = get_single_device('Nexus-6P:1008')
    # d3 = get_single_device('AirBeam:001896105F77')
