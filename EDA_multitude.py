import pandas as pd
import numpy as np
import requests
import json
import folium
from folium import plugins
from time import time
# import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import cPickle as pickle
import os
import geopy
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation
from sklearn.cross_validation import KFold
from geopy.distance import vincenty
import multiprocessing
import seaborn as sbs
# import plotly.plotly as py
# import plotly.tools as tls
from subprocess import call


def get_data():
    '''
    return event dataframe merged with event_readings
    '''
    event_cols = ['event_id', 'weird_hash_thing', 'two', 'device_id', 'zero', 'NaNs', 'file_path?',
    '13-digit_int', 'lat', 'lon', 'device_type', 'two_again', 'device_type_again','device_id_again', 'zero_again', 'url_session_num', '10-digit-int','int_2_to_1101']
    event = pd.read_csv("eventgt.csv", names = event_cols)
    event.set_index('event_id', inplace =True)
    event['timestamp'] = pd.to_datetime(event['10-digit-int'] - 4 * 60 * 60, unit = 's')#, format = "%Y-%m-%d")
    event['day'] = [x.day for x in event.timestamp]
    event['month'] = [x.month for x in event.timestamp]
    event['year'] = [x.year for x in event.timestamp]
    event.drop(['two_again', 'device_type_again', 'device_id_again',
    'zero_again', '10-digit-int','url_session_num','int_2_to_1101','file_path?',
    'NaNs','zero','weird_hash_thing','two','13-digit_int'], axis=1, inplace = True)
    event['coord'] = zip(event.lat,event.lon)

    readings_cols = ['event_id','value','measurment']
    readings = pd.read_csv("event_readingsgt.csv", names = readings_cols)
    readings.set_index('event_id', inplace = True)
    readings = readings.pivot(columns = 'measurment', values = 'value')
    event = event.join(readings).sort_values('timestamp')
    print "Data has been read in the event data frame is {} rows with {} collumns".format(event.shape[0], event.shape[1])
    return event

def find_major_cities(event):
    '''
    INPUT: df: a dataframe with lat and lon columns
    OUTPUT: a list of (lat,lon) coordinates that could not be looked up

    The events of the input dataframe will be grouped by lat and lon using DBSCAN
    and assigned to a county or suburb according to open street maps
    county will be a new column in the dataframe with the string location
    city_center will have the lat and lon for where the center of the city is
    '''
    # n_jobs = multiprocessing.cpu_count()
    db = DBSCAN(eps = .1, min_samples = 100) # eps 1 = 111.2KM, 0.1 = 11.12KM 0.01 = 1.112KM

    stationary_devices = ['AirBeam:001896106892','Terrier-0006667BB11C',
        'AirBeam:0018961061BB','Terrier-0006667BB04A','AirBeam:0018961086FB',
        'Terrier-0006667BB11A']
    data = event[[x not in stationary_devices for x in event.device_id]][['lat','lon']]
    labels = db.fit_predict(data)
    labels = np.expand_dims(labels, axis = 1)
    data['county'] = labels
    geocoder = geopy.geocoders.Nominatim()
    county_mapper = {}
    county_centroid_mapper = {}
    exceptions = []
    for label in np.unique(labels):
        X = data[data.county == label][['lat','lon']]
        lat, lon = X.mean()
        location = geocoder.reverse((lat,lon))
        try:    # Works for most countries
            county_mapper[label] = location.raw['address']['county']
            county_centroid_mapper[label] = (lat,lon)
        except:
            try:    # UK and CA don't have county so I used city instead
                county_mapper[label] = location.raw['address']['city']
                county_centroid_mapper[label] = (lat,lon)
            except:
                try:    # Algeria uses states
                    county_mapper[label] = location.raw['address']['state']
                except:
                    try:    # Australia uses hamlets
                        county_mapper[label] = location.raw['address']['hamlet']
                    except:# Sweeden and Romania are in unicode so this is done "by hand"
                        if (lat,lon) == (37.465285558094642, 126.95470294348861):
                            county_mapper[label] = 'Sweeden'
                            county_centroid_mapper[label] = (lat,lon)
                        elif (lat,lon) == (44.436265005828062, 26.092086128858508):
                            county_mapper[label] = 'Romania'
                            county_centroid_mapper[label] = (lat,lon)
                        elif (lat, lon) == (24.324210072, -3.24871167471):
                            county_mapper[lable] = 'Algeria'
                            county_centroid_mapper[label] = (lat,lon)
                        else:   # One set has weid behavior 923 events over 52 devices
                            exceptions.append((lat,lon))
                            county_mapper[label] = (lat,lon)
                            county_centroid_mapper[label] = (lat,lon)
    data['city_center'] = data.county.map(county_centroid_mapper)
    data.county = data.county.map(county_mapper)
    event = event.join(data[['city_center','county']])

    static_county_mapper = {}
    static_centroid_mapper = {}
    for device in stationary_devices:
        dat = event[event.device_id == device]['coord'].to_frame()
        coords = dat.coord.unique()
        for coord in coords:
            min_dist = 1000
            for cent, city in zip(data.city_center.unique(),data.county.unique()):
                if vincenty(coord,cent).km < min_dist:
                    min_dist = vincenty(coord,cent).km
                    static_centroid_mapper[device] = cent
                    static_county_mapper[device] = city
    event['county_temp'] = event.device_id.map(static_county_mapper)
    event.loc[event.county_temp.notnull(),'county'] = event.county_temp
    event['city_center_temp'] = event.device_id.map(static_centroid_mapper)
    event.loc[event.city_center_temp.notnull(),'city_center'] = event.city_center_temp
    event.drop(['county_temp','city_center_temp'],axis = 1, inplace = True)
    return event, exceptions

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

def transform_data_for_contour_map(df, poll, dec):
    data = df[['lat','lon',poll]].copy()
    for col in data:
        data[col] = data[col].apply(lambda x: round(x,dec))
    x_vals = data.iloc[:,0].unique()
    x_vals.sort()
    y_vals = data.iloc[:,1].unique()
    y_vals.sort()
    mat = np.full((len(x_vals),len(y_vals)),10)
    n = x_vals.size
    for ix, x in enumerate(x_):
        for iy, y in enumerate(y_vals):
            try:
                mat[ix,iy] = data[data.iloc[:,0] == x][data.iloc[:,1] == y].iloc[:,2].mean()
            except:
                pass
        print ix, "of ", n

    return x_vals, y_vals, mat

def save_contour_map(df, poll, fpath, dec = 2):
    x_vals, y_vals, mat = transform_data_for_contour_map(df, poll, dec)
    trace1 = {
        "z": mat,
        "x" : x_vals,
        "y" : y_vals,
        'connectgaps':True,
        'type':'contour',
        'showscale':True,
        'opacity':.5
        }
    fig = tls.make_subplots(1,1)
    fig.append_trace(trace1,1,1)
    py.image.save_as(fig, filename = fpath + ".png")
    return mat



def heat_map(df, pollutant = 'PM25'):
    lat, lon = df[['lat','lon']].mean()
    data = df[['lat','lon',pollutant]].values.tolist()
    mapa = folium.Map((lat,lon), tiles='cartodbpositron',zoom_start = 12)
    mapa.add_children(plugins.HeatMap(data))
    mapa.save('test.html')

def find_corroborators(list_to_map, meters = 50, seconds = 300, pollutant = 'PM25'):
    '''
    INPUT: list of tuples [(device_id, coord, timestamp, pollutant_level), ...]
        to be compared.
    OUTPUT: list of format [((device_id,(device_id, distance, time_diff, poll_diff)), ...]
    '''
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    return pool.map(corrobs,list_to_map)

def corrobs(item, meters = 50, seconds = 300, pollutant = 'PM25'):
    '''
    INPUT: item: a tuple of the form (device_id, coord, timestamp, pollutant_level)
        meters: the minimum distance between two points for them to be considered
        seconds: the minimum time in seconds between two points for them to be considered
        pollutant: the type of pollutant to be compared
    OUTPUT: list of the form list of format [((device_id,(device_id, distance, time_diff, poll_diff)), ...]
    '''
    corroborators = []
    for idx in range(len(ids)):
        if item[0] != ids[idx] and \
            vincenty(item[1],coords[idx]).m <= meters and \
            np.abs((item[2] - times[idx]).total_seconds()) <= seconds:

            corroborators.append((item[0],(ids[idx],vincenty(item[1],coords[idx]).m,
                (item[2] - times[idx]).total_seconds(), item[3] - polls[idx])))
    return corroborators


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

def get_single_date(date, df):
    '''
    INPUT: date: string of the form "yyyy-mm-dd"
           df: dataframe with timestamp field labeled "timestamp"
    OUTPUT a sebset of df where timestamp.date() == date
    '''
    return df[[x.date() == pd.to_datetime(date).date() for x in df.timestamp]]

def export_for_visit(df, city, poll, date):
     data = df[['lon','lat',poll]].dropna()
     data.columns = ["X","Y",poll]
     data['Z'] = np.zeros(data.shape[0])
     data = data[["X","Y","Z",poll]]
     if not os.path.exists('data/data_for_visit/{}/{}'.format(city.replace(' ','_'),poll,poll,date)):
         os.makedirs('data/data_for_visit/{}/{}'.format(city.replace(' ','_'),poll))
     if not os.path.exists('plots/visit/{}/{}'.format(city.replace(' ','_'),poll,poll,date)):
         os.makedirs('plots/visit/{}/{}'.format(city.replace(' ','_'),poll))
     data.to_csv('data/data_for_visit/{}/{}/{}_{}.3D'.format(city.replace(' ','_'),poll,poll,date),sep = ' ', index = False)


def make_visit_plots(cities, pollutants):
    '''
    INPUT cities: a dictionary with {"city":{"df":city_df}} structure
          pollutants: a list of strings corresponding to collumn names of city_df
    OUTPUT None
    creates a .png image for each date in city_df for each pollutant using visit
    '''
    path = os.getenv('PATH')
    path += ":/Applications/VisIt.app/Contents/Resources/bin/"
    n = 1
    for city in cities:
        min_lon = str(round(cities[city]['min_lon'],4))
        max_lon = str(round(cities[city]['max_lon'],4))
        min_lat = str(round(cities[city]['min_lat'],4))
        max_lat = str(round(cities[city]['max_lat'],4))
        for poll in pollutants:
            for date in cities[city]['dates']:
                export_for_visit(get_single_date(date,cities[city]['df']), city, poll, date)
                call(['visit','-cli','-s','visit_script.py',city, poll, min_lon, max_lon, min_lat, max_lat, date], env = {'PATH' : path})
                print "You've made %i plots" % n
                n+=1

def make_city_dictionary(df):
    stationary_devices = ['AirBeam:001896106892','Terrier-0006667BB11C',
            'AirBeam:0018961061BB','Terrier-0006667BB04A','AirBeam:0018961086FB',
            'Terrier-0006667BB11A']
    cities = {}
    for city in df.county.unique():
        cities[city] = {}
        cities[city]['df'] = df[df['county'] == city]
    return cities

def calculate_city_stats(metrics, cities):
    for city in cities:
        for met in metrics:
            cities[city]['avg_{}'.format(met)] = cities[city]['df'][met].mean()
        cities[city]['dates'] =np.unique([x.date().isoformat() for x in cities[city]['df']['timestamp'].sort_values()])# if get_single_date(x.date().isoformat(),cities[city]['df']).shape[0] > 60])
        cities[city]['min_lon'] = cities[city]['df']['lon'].min()
        cities[city]['max_lon'] = cities[city]['df']['lon'].max()
        cities[city]['min_lat'] = cities[city]['df']['lat'].min()
        cities[city]['max_lat'] = cities[city]['df']['lat'].max()
        # cities[city]['pollutants'] =

def format_city_names(event):
    names = event.county.unique()
    unicode_names = []
    for name in names:
        try:
            str(name)
        except:
            unicode_names.append(name)
    names_dict = {u'\uc11c\uc6b8':'Seoul',
                  u'R\u012bga':"Riga",
                  u'Bucure\u0219ti':"Bucuresti",
                  u'D\xe2mbovi\u021ba':"Dambovita",
                  u'Komuna e Prishtin\xebs':"Komuna e Prishtines",
                   u'Constan\u021ba':"Constanta"}
    event['temp_county'] = event.county.map(names_dict)
    event.loc[event.temp_county.notnull(),'county'] = event.temp_county
    event.drop('temp_county',axis = 1, inplace = True)

def make_hour_averages_plot(ny, poll = 'PM25'):
    df = ny[['timestamp','PM25']].dropna()
    df.set_index('timestamp', inplace = True)
    df = df.resample('1H').mean()

    # time_index = [x.strftime('%-m/%-d/%Y %I:%M %p') for x in ny.timestamp]
    gs = pd.read_excel('data/NYSDEC_hourly_070116_071516.xls', skiprows=[0,1,3,4],
        convert_float=False, skip_footer=8, na_values=['InVld','<Samp','Maintain','Audit'])
    gs.columns = [x.replace(' ',"_").lower().replace('&_','') for x in gs.columns]
    gs.set_index('date_time', inplace = True)
    gs.index = gs.index.to_datetime()
    start = gs.index.min()
    end = gs.index.max()

    data = df[(df.index >= start) & (df.index <= end)]
    data['gold_standard'] = gs.mean(axis = 1)
    data.columns = ['Mobile Sensors','Gold Standard']
    ax = data.plot()
    label = ax.set_ylabel("PM 2.5 ug/m^3", fontsize = 20)
    title = ax.set_title("Hour Avg Mobile Sensors vs. Gold Standard in NYC", fontsize = 20)
    plt.show()

def format_visit_plots():
    f_path1 = '/Users/jakenoble/DSI/multiplot/plots/visIt'
    f_path2 = '/Users/jakenoble/DSI/multiplot/images'
    n = 1
    for city in ['New York County']:
        for poll in ['PM25','CO','CO2','NO']:
            for date in cities[city]['dates']:
                call(['open','{}/{}/{}/{}_{}_0000.png'.format(f_path1,city.replace(' ','_'),poll,poll,date)])
                if not os.path.exists('{}/{}/{}'.format(f_path2,city.replace(' ','_'),poll)):
                    os.makedirs('{}/{}/{}'.format(f_path2,city.replace(' ','_'),poll))
                call(['screencapture','-T','1','-R','175,121,645,623', '{}/{}/{}/{}_{}.png'.format(f_path2,city.replace(' ','_'),poll,poll,date)])
                call(['osascript', '-e', 'quit app "Preview"'])
                print "\n made %i plots" %n
                print '{}/{}/{}/{}_{}.png'.format(f_path2,city,poll,poll,date)
                n+=1

if __name__ == '__main__':
    # event =  get_data()
    # event, exceptions = find_major_cities(event)
    # format_city_names(event)
    # event.to_pickle('event.pkl')
    event = pickle.load(open('event.pkl','rb'))

    # Make dictionary of city: city_df pairs
    cities = make_city_dictionary(event)

    metrics = ['BaroP','CO','CO2','NO','PM25','Rh','noise','lat','lon']
    calculate_city_stats(metrics, cities)
    # city_center = [['New_York_County',(40.73,-73.895)]]
    # make_hour_averages_plot(cities['New York County']['df'])
    # make_visit_plots(cities = cities, pollutants = ['CO','CO2','NO','PM25'])
    format_visit_plots()
