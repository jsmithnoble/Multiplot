import cPickle as pickle
import pandas as pd
import numpy as np
import folium
from folium import plugins


def heat_map(df, pollutant = 'PM25'):
    mask = np.ma.masked_invalid(ny[pollutant])

    data = df[~mask.mask][['lat','lon',pollutant]]
    lat, lon, poll = data.mean()
    print "the averages are :{}".format((lat, lon, poll))
    data[pollutant] = data[pollutant] / 35 #data[pollutant].max()
    data = np.vstack((data,[40.7026,-74.0307,1000]))
    gradient = {0:'lime',.5:'blue',1:'red'}
    mapa = folium.Map((lat,lon), tiles='cartodbpositron',zoom_start = 12)
    mapa.add_children(plugins.HeatMap(data.tolist(), min_opacity = 0, gradient = gradient))
    mapa.save('test.html')

naaqs = {'PM25':{'1 day':35, '1 year':12},
    'CO':{'1 hr':35, '8 hr':9},
    'CO2':{'8 hr':5000}}

if __name__ == '__main__':
    event = pickle.load(open('event.pkl','rb'))
    ny = event[event.county == "New York County"]
    heat_map(ny)
