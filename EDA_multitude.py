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

def get_data():
    '''
    return event and event_readings data frames
    '''
    event_cols = ['event_id', 'weird_hash_thing', 'two', 'device_id', 'zero', 'NaNs', 'file_path?',
    '13-digit_int', 'lat', 'long', 'device_type', 'two_again', 'device_type_again','device_id_again', 'zero_again', 'url_session_num', '10-digit-int','int_2_to_1101']
    event = pd.read_csv("event.csv", names = event_cols)
    event['in_ny'] = (event['lat']>= 40.47739) & (event['lat']<=40.917577) & (event['long'] >= -74.25909) & (event['long'] <= -73.700009)
    event.set_index('event_id', inplace =True)
    event['timestamp'] = pd.to_datetime(event['10-digit-int'] - 4 * 60 * 60, unit = 's')#, format = "%Y-%m-%d")
    event = event.drop(['two_again', 'device_type_again', 'device_id_again', 'zero_again', '10-digit-int','url_session_num','int_2_to_1101','file_path?','NaNs','zero','weird_hash_thing','two','13-digit_int'], axis=1)
    readings_cols = ['event_id','value','measurment']
    readings = pd.read_csv("event_readings.csv", names = readings_cols)
    readings.set_index('event_id', inplace = True)
    readings = readings.pivot(columns = 'measurment', values = 'value')
    return event.join(readings).sort_values('timestamp')

def is_in_ny(data):
    '''
    data = DataFrame with lat long as column names

    Returns Series object specifying 1 for if the point is in NYC boundry box
    Note to self: this would be improved if it could specify stricter shapes or neighborhoods
    '''
    lat_boundry = (40.477399, 40.917577)
    lon_boundry = (-74.25909, -73.700009)
    data['in_ny'] = (data['lat']>= 40.47739) & (data['lat']<=40.917577) & (data['long'] >= -74.25909) & (data['long'] <= -73.700009)

    return data


def get_location(coords):
    '''
    INPUT: list of tuples containing (latitude, longitude)
    OUTPUT: list of strings containing response from openstreetmap

    This function uses a reverse geocode lookup from https://nominatim.openstreetmap.org
    '''
    start = time()
    loc = []
    ct = 0
    for lat, lon in coords:
        if ct % 100 == 0:
            print "on event {}".format(ct)
        ct += 1
        try:
            r = requests.get("https://nominatim.openstreetmap.org/reverse.php?format=html&lat={}&lon={}&zoom=".format(lat,lon))
            soup = BeautifulSoup(r.content, 'html.parser')
            loc.append(soup.find_all(class_="name")[0].text)
        except:
            loc.append("not available")
    print "completed in {} seconds".format(time() - start)
    return loc

def get_city_state_country(coords):
    '''
    mostly useless because google api is stingy
    '''
    gmaps_key = os.environ.get("gmaps_key")
    json_decoder = json.JSONDecoder()
    cities = []
    states = []
    countries = []
    counter = 0
    for lat, lon in coords:
        if counter % 100 == 0:
            print "on user {}".format(counter)
        counter += 1
        r = requests.get("https://maps.googleapis.com/maps/api/geocode/json?latlng={},{}&key={}".format(lat,lon, gmaps_key))
        content = json_decoder.decode(r.content)
        try:
            cities.append(str(content['results'][0]['address_components'][3]['long_name'].encode('ascii',errors = 'ignore')))
        except:
            cities.append('not available')
        try:
            states.append(str(content['results'][0]['address_components'][4]['long_name'].encode('ascii',errors = 'ignore')))
        except:
            states.append('not available')
        try:
            countries.append(str(content['results'][0]['address_components'][5]['long_name']))
        except:
            countries.append('not available')
    pickle.dump(cities, open('cities.pkl', 'wb'))
    pickle.dump(states, open('states.pkl', 'wb'))
    pickle.dump(countries, open('countries.pkl', 'wb'))
    return cities, states, countries

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
    lon = data.long.mean()
    m = folium.Map((lat,lon), zoom_start = 12)
    for ride in rides:
        temp = data[data.ride == ride]
        coords = zip(temp.lat,temp.long)
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

# def

# terminal command to connect to db and download event table and convert to csv
# psql -h 104.196.143.165 -U multitude -d d2c_api_prod -p 5432 -o event.csv -c 'select * from event;' -P format=unaligned -P tuples_only -P fieldsep=\,
# psql -h 104.196.143.165 -U multitude -d d2c_api_prod -p 5432 -o event_meta_data.csv -c 'select * from event_meta_data;' -P format=unaligned -P tuples_only -P fieldsep=\,
# psql -h 104.196.143.165 -U multitude -d d2c_api_prod -p 5432 -o event_readings.csv -c 'select * from event_readings;' -P format=unaligned -P tuples_only -P fieldsep=\,
# psql -h <host_ip> -U <username> -d <database> -p <port> -o <output_file_path> -c '<query>' -P format=unaligned -P tuples_only -P fieldsep=\,

if __name__ == '__main__':
    event = get_data()
    # coords = zip(event.lat,event.long)
    # locations = get_location(coords)
    # event['location'] = locations
    pickle.dump(event, open('event.pkl', 'wb'))
    d1 = get_single_device('SAMSUNG-SM-G920A:1083')
    ny = event[event['in_ny'] == True]
    # lat = d1.lat.mean()
    # lon = d1.long.mean()
    # days = d1.timestamp.dt.date.unique()
    # for n, day in enumerate(days):
    #     m = folium.Map((lat,lon), zoom_start = 13)
    #     temp = d1[d1.timestamp.dt.date == day]
    #     coords = zip(temp.lat,temp.long)
    #     folium.PolyLine(coords).add_to(m)
    #     folium.Marker(coords[0], popup = 'Start').add_to(m)
    #     folium.Marker(coords[-1], popup = "End").add_to(m)
    # m.save('plots/devices/{}.html'.format(device_id))
    """Same user for 'Nexus-6P:1008' and 'AirBeam:001896105F77'"""

    # plot_single_device('Nexus-6P:1008')
    # plot_single_device('AirBeam:001896105F77')
    # d2 = get_single_device('Nexus-6P:1008')
    # d3 = get_single_device('AirBeam:001896105F77')
