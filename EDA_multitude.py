import pandas as pd
import requests
import json
import folium
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
    event = event.drop(['two_again', 'device_type_again', 'device_id_again', 'zero_again'], axis=1)
    event['timestamp'] = pd.to_datetime(event['10-digit-int'] - 4 * 60 * 60, unit = 's')#, format = "%Y-%m-%d")
    readings_cols = ['event_id','value','measurment']
    readings = pd.read_csv("event_readings.csv", names = readings_cols)
    readings.set_index('event_id', inplace = True)
    readings = readings.pivot(columns = 'measurment', values = 'value')
    return event, readings

def get_single_device(device_id, data):
    data = data[data['device_id'] == device_id].sort('timestamp')
    time_delta = [0]
    for idx, time in enumerate(data.timestamp):
        if idx == 0:
            pass
        else:
            time_delta.append(time - data.timestamp.iloc[idx - 1])

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



"""46140 events
141 device_id shows 141 different devices --I think I can only keep AirBeams (18,712) and Terriers (1,879)
device_type says all devices are either AirBeams or Terriers
"""

def get_city_state_country(coords):
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

def plot_single_device(device_id):

# terminal command to connect to db and download event table and convert to csv
# psql -h 104.196.143.165 -U multitude -d d2c_api_prod -p 5432 -o event.csv -c 'select * from event;' -P format=unaligned -P tuples_only -P fieldsep=\,
# psql -h 104.196.143.165 -U multitude -d d2c_api_prod -p 5432 -o event_meta_data.csv -c 'select * from event_meta_data;' -P format=unaligned -P tuples_only -P fieldsep=\,
# psql -h 104.196.143.165 -U multitude -d d2c_api_prod -p 5432 -o event_readings.csv -c 'select * from event_readings;' -P format=unaligned -P tuples_only -P fieldsep=\,
# psql -h <host_ip> -U <username> -d <database> -p <port> -o <output_file_path> -c '<query>' -P format=unaligned -P tuples_only -P fieldsep=\,

if __name__ == '__main__':
    event, readings = get_data()
    coords = zip(event.lat,event.long)
    # city, state, country = get_city_state_country(coords)
    d1 = get_single_device('SGH-M919:1070',event)

    lat = d1.lat.mean()
    lon = d1.long.mean()
    days = d1.timestamp.dt.date.unique()
    for n, day in enumerate(days):
        m = folium.Map((lat,lon), zoom_start = 13)
        temp = d1[d1.timestamp.dt.date == day]
        coords = zip(temp.lat,temp.long)
        folium.PolyLine(coords).add_to(m)
        folium.Marker(coords[0], popup = 'Start').add_to(m)
        folium.Marker(coords[-1], popup = "End").add_to(m)
    m.save('plots/devices/{}.html'.format(device_id))
