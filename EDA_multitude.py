import pandas as pd
import numpy as np
import requests
import json
import folium
from folium import plugins
from time import time
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import cPickle as pickle
import os
import geopy
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation
from geopy.distance import vincenty


def get_data():
    '''
    return event dataframe merged with event_readings
    '''
    event_cols = ['event_id', 'weird_hash_thing', 'two', 'device_id', 'zero', 'NaNs', 'file_path?',
    '13-digit_int', 'lat', 'lon', 'device_type', 'two_again', 'device_type_again','device_id_again', 'zero_again', 'url_session_num', '10-digit-int','int_2_to_1101']
    event = pd.read_csv("event.csv", names = event_cols)
    event.set_index('event_id', inplace =True)
    event['timestamp'] = pd.to_datetime(event['10-digit-int'] - 4 * 60 * 60, unit = 's')#, format = "%Y-%m-%d")
    event.drop(['two_again', 'device_type_again', 'device_id_again',
    'zero_again', '10-digit-int','url_session_num','int_2_to_1101','file_path?',
    'NaNs','zero','weird_hash_thing','two','13-digit_int'], axis=1, inplace = True)
    event['coord'] = zip(event.lat,event.lon)

    readings_cols = ['event_id','value','measurment']
    readings = pd.read_csv("event_readings.csv", names = readings_cols)
    readings.set_index('event_id', inplace = True)
    readings = readings.pivot(columns = 'measurment', values = 'value')
    return event.join(readings).sort_values('timestamp')

def get_single_device(device_id, ride_break = 15):
    '''
    INPUT: devide_id: a string corresponding to a device_id in event
            ride_break: number of minutes to define a ride break
    OUTPUT: a dataframe with only on device_id and new features engineered

    New features:
    time_since_last_pt is the duration sinece the last point was observed,
    used to define a new ride.
    ride is a integer which represents which ride the event is associated with.
    '''
    data = event[event['device_id'] == device_id].sort_values('timestamp')
    time_delta = [pd.tslib.Timedelta(1e3, unit = 'm')]
    for idx, time in enumerate(data.timestamp):
        if idx == 0:
            pass
        else:
            time_delta.append(time - data.timestamp.iloc[idx - 1])
    data['time_since_last_pt'] = time_delta
    ride_break = pd.tslib.Timedelta(ride_break, unit = 'm')
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
    m.save("plots/devices/{}.html".format(device_id))

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
    '''
    INPUT: df: a dataframe with lat and lon columns
    OUTPUT: a list of (lat,lon) coordinates that could not be looked up

    The events of the input dataframe will be grouped by lat and lon using DBSCAN
    and assigned to a county or suburb according to open street maps
    county will be a new column in the dataframe with the string location
    city_center will have the lat and lon for where the center of the city is
    '''
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



def find_city_centroids(df, n_clusters = 20, pollutant = 'PM25'):
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
    X.drop('timestamp', axis = 1, inplace = True)
    X = (X - X.mean())/X.std()

    # estimator = KMeans(n_clusters = n_clusters)
    # estimator = DBSCAN(eps = 1e-10, min_samples = 10)
    estimator = AffinityPropagation(damping = .6, max_iter = 200)

    labels = estimator.fit_predict(X)
    # labels = np.expand_dims(labels, axis = 1)
    X['label'] = labels
    X['centroid'] = labels
    X['centroid_time'] = labels
    df = df.join(X[['label','centroid','centroid_time']])
    centroids = {}
    time = {}
    for label in np.unique(labels):
        temp = df[df.centroid == label][['lat', 'lon','timestamp']]
        lat, lon = temp[['lat','lon']].mean()
        centroids[label] = (lat, lon)
        time[label] = pd.to_datetime(temp.timestamp.apply(lambda x: x.value).mean())
    df['centroid'] = df['centroid'].map(centroids)
    df['centroid_time'] = df['centroid_time'].map(time)

    df['coord'] = zip(df.lat,df.lon)
    v_vincenty = np.vectorize(vincenty)
    df['dist_to_centroid_m'] = v_vincenty(df.coord,df.centroid)
    df.dist_to_centroid_m = df.dist_to_centroid_m.apply(lambda x: x.m)

    def time_delta(x,y):
        try:
            return x-y
        except:
            return None
    v_time_delta = np.vectorize(time_delta)
    df['time_to_centroid_m'] = v_time_delta(df.timestamp, df.centroid_time)
    df.time_to_centroid_m = df.time_to_centroid_m.apply(lambda x: x.total_seconds()/60)
    df['time_to_centroid_m_abs'] = df.time_to_centroid_m.abs()
    return df, time

def find_corroborators(city_df, dist = 50, time = 300, pollutant = 'PM25'):
    mask = np.ma.masked_invalid(city_df[pollutant])
    df = city_df[~mask.mask]
    dic_dex = {}
    for event in df.index:
        e = df[df.index == event]
        for other in df.index:
            o = df[df.index == other]
            if (e.device_id.iloc[0] != o.device_id.iloc[0]) and \
               (vincenty(e.coord.iloc[0],o.coord.iloc[0]).m < dist) and \
               (np.abs((e.timestamp.iloc[0] - o.timestamp.iloc[0]).total_seconds()) < time):
               if event in dic_dex.keys():
                   dic_dex[event].append((other,
                        vincenty(e.coord.iloc[0], o.coord.iloc[0]),
                        (e.timestamp.iloc[0] - o.timestamp.iloc[0]).total_seconds(),
                        e[pollutant].iloc[0] - o[pollutant].iloc[0]))
               else:
                   dic_dex[event] = [(other,
                        vincenty(e.coord.iloc[0], o.coord.iloc[0]),
                        (e.timestamp.iloc[0] - o.timestamp.iloc[0]).total_seconds(),
                        e[pollutant].iloc[0] - o[pollutant].iloc[0])]
    return dic_dex

def heat_map(df, pollutant = 'PM25'):
    lat, lon = df[['lat','lon']].mean()
    data = df[['lat','lon',pollutant]].values.tolist()
    mapa = folium.Map((lat,lon), tiles='cartodbpositron',zoom_start = 12)
    mapa.add_children(plugins.HeatMap(data))
    mapa.save('test.html')

# terminal command to connect to db and download event table and convert to csv
# psql -h 104.196.143.165 -U multitude -d d2c_api_prod -p 5432 -o event.csv -c 'select * from event;' -P format=unaligned -P tuples_only -P fieldsep=\,
# psql -h 104.196.143.165 -U multitude -d d2c_api_prod -p 5432 -o event_meta_data.csv -c 'select * from event_meta_data;' -P format=unaligned -P tuples_only -P fieldsep=\,
# psql -h 104.196.143.165 -U multitude -d d2c_api_prod -p 5432 -o event_readings.csv -c 'select * from event_readings;' -P format=unaligned -P tuples_only -P fieldsep=\,
# psql -h <host_ip> -U <username> -d <database> -p <port> -o <output_file_path> -c '<query>' -P format=unaligned -P tuples_only -P fieldsep=\,
terriers = ['Terrier-B110','Terrier-B11E','Terrier-B037','Terrier-B04F',
    'Terrier-B115','Terrier-B122','Terrier-B12D','Terrier-B124','Terrier-B035',
    'Terrier-B051','Terrier-B02F','Terrier-B125','Terrier-B038','Terrier-B121',
    'Terrier-B033','Terrier-B054','Terrier-B05B','Terrier-B02E','Terrier-B004',
    'Terrier-B03C','Terrier-B121','Terrier-B054']
airbeams = ['AirBeam-0018961059B7','AirBeam-001896105985','AirBeam-001896105F53',
    'AirBeam-001896106134','AirBeam-001896105574','AirBeam-00189610687D',
    'AirBeam-001896105561','AirBeam-0018961051B6','AirBeam-001896108705',
    'AirBeam-00189610553A','AirBeam-001896106D2D','AirBeam-001896105550',
    'AirBeam-0018961086F4','AirBeam-00189610685A','AirBeam-001896105556',
    'AirBeam-001896105F32','AirBeam-0018961086C4','AirBeam-001896106863',
    'AirBeam-00189610688D','AirBeam-001896014191','AirBeam-00189610685A',
    'AirBeam-001896105F32']
users = ['MB005','MB001','MB002','MB008','MB010','MB012','MB014','MB016','MB011',
    'MB004','MB007','MB009','MB003','MB006','MB013','MB015','MB017','MB018',
    'MB019','MB020','MB021','MB022']
paired_devices = pd.DataFrame(np.vstack((airbeams,terriers)).T,index = users, columns = ['AirBeam','Terrier'])
if __name__ == '__main__':
    # event = get_data()
    # exceptions = find_major_cities(event)
    # event.to_pickle('event.pkl')
    event = pickle.load(open('event.pkl','rb'))

    # d1 = get_single_device('SAMSUNG-SM-G920A:1083')
    ny = event[event['county'] == "New York County"]
    # ny, time = find_city_centroids(ny)
    # # df = ny[['coord','centroid','timestamp','centroid_time','dist_to_centroid_m','time_to_centroid_s']]
    mask = np.ma.masked_invalid(ny['PM25'])
    ny25 = ny[~mask.mask]


    naaqs = {'PM25':{'1 day':35, '1 year':12},
            'CO':{'1 hr':35, '8 hr':9},
            'CO2':{'8 hr':5000}}
    # dic_dex = {}
    # for event in ny25.index:
    #     e = ny25[ny25.index == event]
    #     for other in ny25.index:
    #         o = ny25[ny25.index == other]
    #         if (e.device_id.iloc[0] != o.device_id.iloc[0]) and \
    #            (vincenty(e.coord.iloc[0],o.coord.iloc[0]).m < 50) and \
    #            (np.abs((e.timestamp.iloc[0] - o.timestamp.iloc[0]).total_seconds()) < 300):
    #            if event in dic_dex.keys():
    #                dic_dex[event].append((other,
    #                     vincenty(e.coord.iloc[0], o.coord.iloc[0]),
    #                     (e.timestamp.iloc[0] - o.timestamp.iloc[0]).total_seconds(),
    #                     e.pm25.iloc[0] - o.PM25.iloc[0]))
    #            else:
    #                dic_dex[event] = [(other,
    #                     vincenty(e.coord.iloc[0], o.coord.iloc[0]),
    #                     (e.timestamp.iloc[0] - o.timestamp.iloc[0]).total_seconds(),
    #                     e.pm25.iloc[0] - o.PM25.iloc[0])]

    # dic_diff = {}
    # for event in dic_dex.iterkeys():
    #     e = ny25[ny25.index == event]
    #     for other in dic_dex[event]:
    #         o = ny25[ny25.index == other]
    #         if event in dic_diff.keys():
    #            dic_diff[event].append((other,
    #                 vincenty(e.coord.iloc[0], o.coord.iloc[0]),
    #                 (e.timestamp.iloc[0] - o.timestamp.iloc[0]).total_seconds(),
    #                 e.pm25.iloc[0] - o.PM25.iloc[0]))
    #         else:
    #             dic_diff[event] = [(other,
    #                  vincenty(e.coord.iloc[0], o.coord.iloc[0]),
    #                  (e.timestamp.iloc[0] - o.timestamp.iloc[0]).total_seconds(),
    #                  e.pm25.iloc[0] - o.PM25.iloc[0])]

    # pickle.dump(dic_dex, open('dic_dex.pkl','wb'))
    # pickle.dump(dic_diff, open('dic_diff.pkl','wb'))

    # labels = ny.label.unique()
    # diff = []
    # for label in labels:
    #     if ny[ny.label == label]['centroid_time'].unique().size > 1:
    #         diff.append(level)


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
