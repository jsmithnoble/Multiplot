from flask import Flask, render_template, request, send_file, send_from_directory
# import matplotlib.pyplot as plt
from random import random
import pandas as pd
import numpy as np
from StringIO import StringIO
import json
from collections import Counter
import cPickle as pickle

def make_city_dictionary(df):
    stationary_devices = ['AirBeam:001896106892','Terrier-0006667BB11C',
            'AirBeam:0018961061BB','Terrier-0006667BB04A','AirBeam:0018961086FB',
            'Terrier-0006667BB11A']
    ['stationary','mobile']
    cities = {}
    for city in df.county.unique():
        cities[city] = {}
        cities[city]['df'] = df[df['county'] == city]
    return cities

def calculate_city_stats(metrics, cities):
    cities_done = 0
    for city in cities:
        metrics_done = 0
        for met in metrics:
            cities[city]['avg_{}'.format(met)] = cities[city]['df'][met].mean()
            metrics_done += 1
        cities[city]['dates'] =np.unique([x.date().isoformat() for x in cities[city]['df']['timestamp'].sort_values()])
        cities[city]['min_lon'] = cities[city]['df']['lon'].min()
        cities[city]['max_lon'] = cities[city]['df']['lon'].max()
        cities[city]['min_lat'] = cities[city]['df']['lat'].min()
        cities[city]['max_lat'] = cities[city]['df']['lat'].max()
        cities_done += 1

event = pickle.load(open('event.pkl','rb'))
cities = make_city_dictionary(event)
metrics = ['BaroP','CO','CO2','NO','PM25','Rh','noise','lat','lon']
calculate_city_stats(metrics, cities)


app = Flask(__name__, static_url_path='')

@app.route('/home')
@app.route('/')
def home():
    return render_template('./index.html')

@app.route('/get_legend')
def get_legend():
    file_name = 'PM25_legend.png'
    return send_from_directory('images', file_name)

@app.route('/get_map')
def get_image():
    city = request.args.get('city')
    poll = request.args.get('poll')
    date = request.args.get('date')
    # file_name = "/Users/jakenoble/Desktop/PM25_nym.png"
    file_name = '{}/{}/{}_{}.png'.format(city.replace(' ','_'),poll,poll, date)
    print file_name,"*"*20
    return send_from_directory('images', file_name)
    # return dend_file(file_name, mimetype='image/png')

@app.route("/nyc")
def ny():
    return render_template("mapplot.html",
        dates=cities['New York County']['dates'],
        poll="PM25",
        min_lon=cities["New York County"]['min_lon'],
        max_lon=cities['New York County']['max_lon'],
        min_lat=cities["New York County"]['min_lat'],
        max_lat=cities['New York County']['max_lat'],
        mean_lat=cities['New York County']['avg_lat'],
        mean_lon=cities['New York County']['avg_lon'],
        avg_NO=round(cities['New York County']['avg_NO'],2),
        avg_CO=round(cities['New York County']['avg_CO'],2),
        avg_CO2=round(cities['New York County']['avg_CO2'],2),
        avg_PM25=round(cities['New York County']['avg_PM25'],2))

@app.route("/boulder")
def boulder():
    return render_template("mapplot.html",
        dates=cities['Boulder County']['dates'],
        poll="PM25",
        min_lon=cities["Boulder County"]['min_lon'],
        max_lon=cities['Boulder County']['max_lon'],
        min_lat=cities["Boulder County"]['min_lat'],
        max_lat=cities['Boulder County']['max_lat'],
        mean_lat=cities['Boulder County']['avg_lat'],
        mean_lon=cities['Boulder County']['avg_lon'],
        avg_NO=round(cities['Boulder County']['avg_NO'],2),
        avg_CO=round(cities['Boulder County']['avg_CO'],2),
        avg_CO2=round(cities['Boulder County']['avg_CO2'],2),
        avg_PM25=round(cities['Boulder County']['avg_PM25'],2))

@app.route("/jeffco")
def jeffco():
    return render_template("mapplot.html",
        dates=cities['Jefferson County']['dates'],
        poll="PM25",
        min_lon=cities["Jefferson County"]['min_lon'],
        max_lon=cities['Jefferson County']['max_lon'],
        min_lat=cities["Jefferson County"]['min_lat'],
        max_lat=cities['Jefferson County']['max_lat'],
        mean_lat=cities['Jefferson County']['avg_lat'],
        mean_lon=cities['Jefferson County']['avg_lon'],
        avg_NO=round(cities['Jefferson County']['avg_NO'],2),
        avg_CO=round(cities['Jefferson County']['avg_CO'],2),
        avg_CO2=round(cities['Jefferson County']['avg_CO2'],2),
        avg_PM25=round(cities['Jefferson County']['avg_PM25'],2))


@app.route("/santiago")
def santiago():
    return render_template("mapplot.html",
        dates=cities['Jefferson County']['dates'],
        poll="PM25",
        min_lon=cities["Provincia de Santiago"]['min_lon'],
        max_lon=cities['Provincia de Santiago']['max_lon'],
        min_lat=cities["Provincia de Santiago"]['min_lat'],
        max_lat=cities['Provincia de Santiago']['max_lat'],
        mean_lat=cities['Provincia de Santiago']['avg_lat'],
        mean_lon=cities['Provincia de Santiago']['avg_lon'],
        avg_NO=round(cities['Provincia de Santiago']['avg_NO'],2),
        avg_CO=round(cities['Provincia de Santiago']['avg_CO'],2),
        avg_CO2=round(cities['Provincia de Santiago']['avg_CO2'],2),
        avg_PM25=round(cities['Provincia de Santiago']['avg_PM25'],2))

@app.route("/riga")
def riga():
    return render_template("mapplot.html",
        dates=cities['Riga']['dates'],
        poll="PM25",
        min_lon=cities["Riga"]['min_lon'],
        max_lon=cities['Riga']['max_lon'],
        min_lat=cities["Riga"]['min_lat'],
        max_lat=cities['Riga']['max_lat'],
        mean_lat=cities['Riga']['avg_lat'],
        mean_lon=cities['Riga']['avg_lon'],
        avg_NO=round(cities['Riga']['avg_NO'],2),
        avg_CO=round(cities['Riga']['avg_CO'],2),
        avg_CO2=round(cities['Riga']['avg_CO2'],2),
        avg_PM25=round(cities['Riga']['avg_PM25'],2))

@app.route("/bangalor")
def bangalor():
    return render_template("mapplot.html",
        dates=cities['Bangalore Urban']['dates'],
        poll="PM25",
        min_lon=cities["Bangalore Urban"]['min_lon'],
        max_lon=cities['Bangalore Urban']['max_lon'],
        min_lat=cities["Bangalore Urban"]['min_lat'],
        max_lat=cities['Bangalore Urban']['max_lat'],
        mean_lat=cities['Bangalore Urban']['avg_lat'],
        mean_lon=cities['Bangalore Urban']['avg_lon'],
        avg_NO=round(cities['Bangalore Urban']['avg_NO'],2),
        avg_CO=round(cities['Bangalore Urban']['avg_CO'],2),
        avg_CO2=round(cities['Bangalore Urban']['avg_CO2'],2),
        avg_PM25=round(cities['Bangalore Urban']['avg_PM25'],2))

@app.route("/vest_agder")
def vest_adger():
    return render_template("mapplot.html",
        dates=cities['Vest-Agder']['dates'],
        poll="PM25",
        min_lon=cities["Vest-Agder"]['min_lon'],
        max_lon=cities['Vest-Agder']['max_lon'],
        min_lat=cities["Vest-Agder"]['min_lat'],
        max_lat=cities['Vest-Agder']['max_lat'],
        mean_lat=cities['Vest-Agder']['avg_lat'],
        mean_lon=cities['Vest-Agder']['avg_lon'],
        avg_NO=round(cities['Vest-Agder']['avg_NO'],2),
        avg_CO=round(cities['Vest-Agder']['avg_CO'],2),
        avg_CO2=round(cities['Vest-Agder']['avg_CO2'],2),
        avg_PM25=round(cities['Vest-Agder']['avg_PM25'],2))

@app.route("/sacremento")
def sacremento():
    return render_template("mapplot.html",
        dates=cities['Sacramento County']['dates'],
        poll="PM25",
        min_lon=cities["Sacramento County"]['min_lon'],
        max_lon=cities['Sacramento County']['max_lon'],
        min_lat=cities["Sacramento County"]['min_lat'],
        max_lat=cities['Sacramento County']['max_lat'],
        mean_lat=cities['Sacramento County']['avg_lat'],
        mean_lon=cities['Sacramento County']['avg_lon'],
        avg_NO=round(cities['Sacramento County']['avg_NO'],2),
        avg_CO=round(cities['Sacramento County']['avg_CO'],2),
        avg_CO2=round(cities['Sacramento County']['avg_CO2'],2),
        avg_PM25=round(cities['Sacramento County']['avg_PM25'],2))

@app.route("/sydeny")
def sydney():
    return render_template("mapplot.html",
        dates=cities['Sydney']['dates'],
        poll="PM25",
        min_lon=cities["Sydney"]['min_lon'],
        max_lon=cities['Sydney']['max_lon'],
        min_lat=cities["Sydney"]['min_lat'],
        max_lat=cities['Sydney']['max_lat'],
        mean_lat=cities['Sydney']['avg_lat'],
        mean_lon=cities['Sydney']['avg_lon'],
        avg_NO=round(cities['Sydney']['avg_NO'],2),
        avg_CO=round(cities['Sydney']['avg_CO'],2),
        avg_CO2=round(cities['Sydney']['avg_CO2'],2),
        avg_PM25=round(cities['Sydney']['avg_PM25'],2))

@app.route("/ramsey")
def ramsey():
    return render_template("mapplot.html",
        dates=cities['Ramsey County']['dates'],
        poll="PM25",
        min_lon=cities["Ramsey County"]['min_lon'],
        max_lon=cities['Ramsey County']['max_lon'],
        min_lat=cities["Ramsey County"]['min_lat'],
        max_lat=cities['Ramsey County']['max_lat'],
        mean_lat=cities['Ramsey County']['avg_lat'],
        mean_lon=cities['Ramsey County']['avg_lon'],
        avg_NO=round(cities['Ramsey County']['avg_NO'],2),
        avg_CO=round(cities['Ramsey County']['avg_CO'],2),
        avg_CO2=round(cities['Ramsey County']['avg_CO2'],2),
        avg_PM25=round(cities['Ramsey County']['avg_PM25'],2))

@app.route("/dallas")
def dallas():
    return render_template("mapplot.html",
        dates=cities['Dallas County']['dates'],
        poll="PM25",
        min_lon=cities["Dallas County"]['min_lon'],
        max_lon=cities['Dallas County']['max_lon'],
        min_lat=cities["Dallas County"]['min_lat'],
        max_lat=cities['Dallas County']['max_lat'],
        mean_lat=cities['Dallas County']['avg_lat'],
        mean_lon=cities['Dallas County']['avg_lon'],
        avg_NO=round(cities['Dallas County']['avg_NO'],2),
        avg_CO=round(cities['Dallas County']['avg_CO'],2),
        avg_CO2=round(cities['Dallas County']['avg_CO2'],2),
        avg_PM25=round(cities['Dallas County']['avg_PM25'],2))

@app.route("/liverpool")
def liverpool():
    return render_template("mapplot.html",
        dates=cities['Liverpool']['dates'],
        poll="PM25",
        min_lon=cities["Liverpool"]['min_lon'],
        max_lon=cities['Liverpool']['max_lon'],
        min_lat=cities["Liverpool"]['min_lat'],
        max_lat=cities['Liverpool']['max_lat'],
        mean_lat=cities['Liverpool']['avg_lat'],
        mean_lon=cities['Liverpool']['avg_lon'],
        avg_NO=round(cities['Liverpool']['avg_NO'],2),
        avg_CO=round(cities['Liverpool']['avg_CO'],2),
        avg_CO2=round(cities['Liverpool']['avg_CO2'],2),
        avg_PM25=round(cities['Liverpool']['avg_PM25'],2))

@app.route("/seoul")
def seoul():
    return render_template("mapplot.html",
        dates=cities['Seoul']['dates'],
        poll="PM25",
        min_lon=cities["Seoul"]['min_lon'],
        max_lon=cities['Seoul']['max_lon'],
        min_lat=cities["Seoul"]['min_lat'],
        max_lat=cities['Seoul']['max_lat'],
        mean_lat=cities['Seoul']['avg_lat'],
        mean_lon=cities['Seoul']['avg_lon'],
        avg_NO=round(cities['Seoul']['avg_NO'],2),
        avg_CO=round(cities['Seoul']['avg_CO'],2),
        avg_CO2=round(cities['Seoul']['avg_CO2'],2),
        avg_PM25=round(cities['Seoul']['avg_PM25'],2))

@app.route("/oswego")
def oswego():
    return render_template("mapplot.html",
        dates=cities['Oswego']['dates'],
        poll="PM25",
        min_lon=cities["Oswego"]['min_lon'],
        max_lon=cities['Oswego']['max_lon'],
        min_lat=cities["Oswego"]['min_lat'],
        max_lat=cities['Oswego']['max_lat'],
        mean_lat=cities['Oswego']['avg_lat'],
        mean_lon=cities['Oswego']['avg_lon'],
        avg_NO=round(cities['Oswego']['avg_NO'],2),
        avg_CO=round(cities['Oswego']['avg_CO'],2),
        avg_CO2=round(cities['Oswego']['avg_CO2'],2),
        avg_PM25=round(cities['Oswego']['avg_PM25'],2))

@app.route("/dambovita")
def dambovita():
    return render_template("mapplot.html",
        dates=cities['Dambovita']['dates'],
        poll="PM25",
        min_lon=cities["Dambovita"]['min_lon'],
        max_lon=cities['Dambovita']['max_lon'],
        min_lat=cities["Dambovita"]['min_lat'],
        max_lat=cities['Dambovita']['max_lat'],
        mean_lat=cities['Dambovita']['avg_lat'],
        mean_lon=cities['Dambovita']['avg_lon'],
        avg_NO=round(cities['Dambovita']['avg_NO'],2),
        avg_CO=round(cities['Dambovita']['avg_CO'],2),
        avg_CO2=round(cities['Dambovita']['avg_CO2'],2),
        avg_PM25=round(cities['Dambovita']['avg_PM25'],2))

@app.route("/Constanta")
def constanta():
    return render_template("mapplot.html",
        dates=cities['Constanta']['dates'],
        poll="PM25",
        min_lon=cities["Constanta"]['min_lon'],
        max_lon=cities['Constanta']['max_lon'],
        min_lat=cities["Constanta"]['min_lat'],
        max_lat=cities['Constanta']['max_lat'],
        mean_lat=cities['Constanta']['avg_lat'],
        mean_lon=cities['Constanta']['avg_lon'],
        avg_NO=round(cities['Constanta']['avg_NO'],2),
        avg_CO=round(cities['Constanta']['avg_CO'],2),
        avg_CO2=round(cities['Constanta']['avg_CO2'],2),
        avg_PM25=round(cities['Constanta']['avg_PM25'],2))

@app.route("/komuna")
def komuna():
    return render_template("mapplot.html",
        dates=cities['Komuna e Prishtines']['dates'],
        poll="PM25",
        min_lon=cities["Komuna e Prishtines"]['min_lon'],
        max_lon=cities['Komuna e Prishtines']['max_lon'],
        min_lat=cities["Komuna e Prishtines"]['min_lat'],
        max_lat=cities['Komuna e Prishtines']['max_lat'],
        mean_lat=cities['Komuna e Prishtines']['avg_lat'],
        mean_lon=cities['Komuna e Prishtines']['avg_lon'],
        avg_NO=round(cities['Komuna e Prishtines']['avg_NO'],2),
        avg_CO=round(cities['Komuna e Prishtines']['avg_CO'],2),
        avg_CO2=round(cities['Komuna e Prishtines']['avg_CO2'],2),
        avg_PM25=round(cities['Komuna e Prishtines']['avg_PM25'],2))

@app.route("/king")
def king():
    return render_template("mapplot.html",
        dates=cities['King County']['dates'],
        poll="PM25",
        min_lon=cities["King County"]['min_lon'],
        max_lon=cities['King County']['max_lon'],
        min_lat=cities["King County"]['min_lat'],
        max_lat=cities['King County']['max_lat'],
        mean_lat=cities['King County']['avg_lat'],
        mean_lon=cities['King County']['avg_lon'],
        avg_NO=round(cities['King County']['avg_NO'],2),
        avg_CO=round(cities['King County']['avg_CO'],2),
        avg_CO2=round(cities['King County']['avg_CO2'],2),
        avg_PM25=round(cities['King County']['avg_PM25'],2))

@app.route("/dubuque")
def dubuque():
    return render_template("mapplot.html",
        dates=cities['Dubuque County']['dates'],
        poll="PM25",
        min_lon=cities["Dubuque County"]['min_lon'],
        max_lon=cities['Dubuque County']['max_lon'],
        min_lat=cities["Dubuque County"]['min_lat'],
        max_lat=cities['Dubuque County']['max_lat'],
        mean_lat=cities['Dubuque County']['avg_lat'],
        mean_lon=cities['Dubuque County']['avg_lon'],
        avg_NO=round(cities['Dubuque County']['avg_NO'],2),
        avg_CO=round(cities['Dubuque County']['avg_CO'],2),
        avg_CO2=round(cities['Dubuque County']['avg_CO2'],2),
        avg_PM25=round(cities['Dubuque County']['avg_PM25'],2))

@app.route("/travis")
def travis():
    return render_template("mapplot.html",
        dates=cities['Travis County']['dates'],
        poll="PM25",
        min_lon=cities["Travis County"]['min_lon'],
        max_lon=cities['Travis County']['max_lon'],
        min_lat=cities["Travis County"]['min_lat'],
        max_lat=cities['Travis County']['max_lat'],
        mean_lat=cities['Travis County']['avg_lat'],
        mean_lon=cities['Travis County']['avg_lon'],
        avg_NO=round(cities['Travis County']['avg_NO'],2),
        avg_CO=round(cities['Travis County']['avg_CO'],2),
        avg_CO2=round(cities['Travis County']['avg_CO2'],2),
        avg_PM25=round(cities['Travis County']['avg_PM25'],2))

@app.route("/johnson")
def johnson():
    return render_template("mapplot.html",
        dates=cities['Johnson County']['dates'],
        poll="PM25",
        min_lon=cities["Johnson County"]['min_lon'],
        max_lon=cities['Johnson County']['max_lon'],
        min_lat=cities["Johnson County"]['min_lat'],
        max_lat=cities['Johnson County']['max_lat'],
        mean_lat=cities['Johnson County']['avg_lat'],
        mean_lon=cities['Johnson County']['avg_lon'],
        avg_NO=round(cities['Johnson County']['avg_NO'],2),
        avg_CO=round(cities['Johnson County']['avg_CO'],2),
        avg_CO2=round(cities['Johnson County']['avg_CO2'],2),
        avg_PM25=round(cities['Johnson County']['avg_PM25'],2))

@app.route("/pima")
def pima():
    return render_template("mapplot.html",
        dates=cities['Pima County']['dates'],
        poll="PM25",
        min_lon=cities["Pima County"]['min_lon'],
        max_lon=cities['Pima County']['max_lon'],
        min_lat=cities["Pima County"]['min_lat'],
        max_lat=cities['Pima County']['max_lat'],
        mean_lat=cities['Pima County']['avg_lat'],
        mean_lon=cities['Pima County']['avg_lon'],
        avg_NO=round(cities['Pima County']['avg_NO'],2),
        avg_CO=round(cities['Pima County']['avg_CO'],2),
        avg_CO2=round(cities['SydPima Countyney']['avg_CO2'],2),
        avg_PM25=round(cities['Pima County']['avg_PM25'],2))

@app.route("/london")
def london():
    return render_template("mapplot.html",
        dates=cities['London']['dates'],
        poll="PM25",
        min_lon=cities["London"]['min_lon'],
        max_lon=cities['London']['max_lon'],
        min_lat=cities["London"]['min_lat'],
        max_lat=cities['London']['max_lat'],
        mean_lat=cities['London']['avg_lat'],
        mean_lon=cities['London']['avg_lon'],
        avg_NO=round(cities['London']['avg_NO'],2),
        avg_CO=round(cities['London']['avg_CO'],2),
        avg_CO2=round(cities['London']['avg_CO2'],2),
        avg_PM25=round(cities['London']['avg_PM25'],2))

@app.route("/grenoble")
def grenoble():
    return render_template("mapplot.html",
        dates=cities['Grenoble']['dates'],
        poll="PM25",
        min_lon=cities["Grenoble"]['min_lon'],
        max_lon=cities['Grenoble']['max_lon'],
        min_lat=cities["Grenoble"]['min_lat'],
        max_lat=cities['Grenoble']['max_lat'],
        mean_lat=cities['Grenoble']['avg_lat'],
        mean_lon=cities['Grenoble']['avg_lon'],
        avg_NO=round(cities['Grenoble']['avg_NO'],2),
        avg_CO=round(cities['Grenoble']['avg_CO'],2),
        avg_CO2=round(cities['Grenoble']['avg_CO2'],2),
        avg_PM25=round(cities['Grenoble']['avg_PM25'],2))

@app.route("/managua")
def managua():
    return render_template("mapplot.html",
        dates=cities['SydnManagua (Municipio)ey']['dates'],
        poll="PM25",
        min_lon=cities["Managua (Municipio)"]['min_lon'],
        max_lon=cities['Managua (Municipio)']['max_lon'],
        min_lat=cities["Managua (Municipio)"]['min_lat'],
        max_lat=cities['Managua (Municipio)']['max_lat'],
        mean_lat=cities['Managua (Municipio)']['avg_lat'],
        mean_lon=cities['Managua (Municipio)']['avg_lon'],
        avg_NO=round(cities['Managua (Municipio)']['avg_NO'],2),
        avg_CO=round(cities['Managua (Municipio)']['avg_CO'],2),
        avg_CO2=round(cities['Managua (Municipio)']['avg_CO2'],2),
        avg_PM25=round(cities['Managua (Municipio)']['avg_PM25'],2))

@app.route("/arlington")
def arlington():
    return render_template("mapplot.html",
        dates=cities['Arlington County']['dates'],
        poll="PM25",
        min_lon=cities["Arlington County"]['min_lon'],
        max_lon=cities['Arlington County']['max_lon'],
        min_lat=cities["Arlington County"]['min_lat'],
        max_lat=cities['Arlington County']['max_lat'],
        mean_lat=cities['Arlington County']['avg_lat'],
        mean_lon=cities['Arlington County']['avg_lon'],
        avg_NO=round(cities['Arlington County']['avg_NO'],2),
        avg_CO=round(cities['Arlington County']['avg_CO'],2),
        avg_CO2=round(cities['Arlington County']['avg_CO2'],2),
        avg_PM25=round(cities['Arlington County']['avg_PM25'],2))

@app.route("/berlin")
def berlin():
    return render_template("mapplot.html",
        dates=cities['Berlin']['dates'],
        poll="PM25",
        min_lon=cities["Berlin"]['min_lon'],
        max_lon=cities['Berlin']['max_lon'],
        min_lat=cities["Berlin"]['min_lat'],
        max_lat=cities['Berlin']['max_lat'],
        mean_lat=cities['Berlin']['avg_lat'],
        mean_lon=cities['Berlin']['avg_lon'],
        avg_NO=round(cities['Berlin']['avg_NO'],2),
        avg_CO=round(cities['Berlin']['avg_CO'],2),
        avg_CO2=round(cities['Berlin']['avg_CO2'],2),
        avg_PM25=round(cities['Berlin']['avg_PM25'],2))

@app.route("/mecklenburg")
def mecklenburg():
    return render_template("mapplot.html",
        dates=cities['Mecklenburg County']['dates'],
        poll="PM25",
        min_lon=cities["Mecklenburg County"]['min_lon'],
        max_lon=cities['Mecklenburg County']['max_lon'],
        min_lat=cities["Mecklenburg County"]['min_lat'],
        max_lat=cities['Mecklenburg County']['max_lat'],
        mean_lat=cities['Mecklenburg County']['avg_lat'],
        mean_lon=cities['Mecklenburg County']['avg_lon'],
        avg_NO=round(cities['Mecklenburg County']['avg_NO'],2),
        avg_CO=round(cities['Mecklenburg County']['avg_CO'],2),
        avg_CO2=round(cities['Mecklenburg County']['avg_CO2'],2),
        avg_PM25=round(cities['Mecklenburg County']['avg_PM25'],2))

@app.route("/la")
def la():
    return render_template("mapplot.html",
        dates=cities['Los Angeles County']['dates'],
        poll="PM25",
        min_lon=cities["Los Angeles County"]['min_lon'],
        max_lon=cities['Los Angeles County']['max_lon'],
        min_lat=cities["Los Angeles County"]['min_lat'],
        max_lat=cities['Los Angeles County']['max_lat'],
        mean_lat=cities['Los Angeles County']['avg_lat'],
        mean_lon=cities['Los Angeles County']['avg_lon'],
        avg_NO=round(cities['Los Angeles County']['avg_NO'],2),
        avg_CO=round(cities['Los Angeles County']['avg_CO'],2),
        avg_CO2=round(cities['Los Angeles County']['avg_CO2'],2),
        avg_PM25=round(cities['Los Angeles County']['avg_PM25'],2))

@app.route("/lillesand")
def lillesand():
    return render_template("mapplot.html",
        dates=cities['Lillesand']['dates'],
        poll="PM25",
        min_lon=cities["Lillesand"]['min_lon'],
        max_lon=cities['Lillesand']['max_lon'],
        min_lat=cities["Lillesand"]['min_lat'],
        max_lat=cities['Lillesand']['max_lat'],
        mean_lat=cities['Lillesand']['avg_lat'],
        mean_lon=cities['Lillesand']['avg_lon'],
        avg_NO=round(cities['Lillesand']['avg_NO'],2),
        avg_CO=round(cities['Lillesand']['avg_CO'],2),
        avg_CO2=round(cities['Lillesand']['avg_CO2'],2),
        avg_PM25=round(cities['Lillesand']['avg_PM25'],2))

@app.route("/alameda")
def alameda():
    return render_template("mapplot.html",
        dates=cities['Alameda County']['dates'],
        poll="PM25",
        min_lon=cities["Alameda County"]['min_lon'],
        max_lon=cities['Alameda County']['max_lon'],
        min_lat=cities["Alameda County"]['min_lat'],
        max_lat=cities['Alameda County']['max_lat'],
        mean_lat=cities['Alameda County']['avg_lat'],
        mean_lon=cities['Alameda County']['avg_lon'],
        avg_NO=round(cities['Alameda County']['avg_NO'],2),
        avg_CO=round(cities['Alameda County']['avg_CO'],2),
        avg_CO2=round(cities['Alameda County']['avg_CO2'],2),
        avg_PM25=round(cities['Alameda County']['avg_PM25'],2))

@app.route("/orange")
def orange():
    return render_template("mapplot.html",
        dates=cities['Orange County']['dates'],
        poll="PM25",
        min_lon=cities["Orange County"]['min_lon'],
        max_lon=cities['Orange County']['max_lon'],
        min_lat=cities["Orange County"]['min_lat'],
        max_lat=cities['Orange County']['max_lat'],
        mean_lat=cities['Orange County']['avg_lat'],
        mean_lon=cities['Orange County']['avg_lon'],
        avg_NO=round(cities['Orange County']['avg_NO'],2),
        avg_CO=round(cities['Orange County']['avg_CO'],2),
        avg_CO2=round(cities['Orange County']['avg_CO2'],2),
        avg_PM25=round(cities['Orange County']['avg_PM25'],2))

@app.route("/miami")
def miami():
    return render_template("mapplot.html",
        dates=cities['Miami-Dade County']['dates'],
        poll="PM25",
        min_lon=cities["Miami-Dade County"]['min_lon'],
        max_lon=cities['Miami-Dade County']['max_lon'],
        min_lat=cities["Miami-Dade County"]['min_lat'],
        max_lat=cities['Miami-Dade County']['max_lat'],
        mean_lat=cities['Miami-Dade County']['avg_lat'],
        mean_lon=cities['Miami-Dade County']['avg_lon'],
        avg_NO=round(cities['Miami-Dade County']['avg_NO'],2),
        avg_CO=round(cities['Miami-Dade County']['avg_CO'],2),
        avg_CO2=round(cities['Miami-Dade County']['avg_CO2'],2),
        avg_PM25=round(cities['Miami-Dade County']['avg_PM25'],2))

@app.route("/shire")
def shire():
    return render_template("mapplot.html",
        dates=cities['Shire of Wellington']['dates'],
        poll="PM25",
        min_lon=cities["Shire of Wellington"]['min_lon'],
        max_lon=cities['Shire of Wellington']['max_lon'],
        min_lat=cities["Shire of Wellington"]['min_lat'],
        max_lat=cities['Shire of Wellington']['max_lat'],
        mean_lat=cities['Shire of Wellington']['avg_lat'],
        mean_lon=cities['Shire of Wellington']['avg_lon'],
        avg_NO=round(cities['Shire of Wellington']['avg_NO'],2),
        avg_CO=round(cities['Shire of Wellington']['avg_CO'],2),
        avg_CO2=round(cities['Shire of Wellington']['avg_CO2'],2),
        avg_PM25=round(cities['Shire of Wellington']['avg_PM25'],2))

@app.route("/edmonton")
def edmonton():
    return render_template("mapplot.html",
        dates=cities['Edmonton']['dates'],
        poll="PM25",
        min_lon=cities["Edmonton"]['min_lon'],
        max_lon=cities['Edmonton']['max_lon'],
        min_lat=cities["Edmonton"]['min_lat'],
        max_lat=cities['Edmonton']['max_lat'],
        mean_lat=cities['Edmonton']['avg_lat'],
        mean_lon=cities['Edmonton']['avg_lon'],
        avg_NO=round(cities['Edmonton']['avg_NO'],2),
        avg_CO=round(cities['Edmonton']['avg_CO'],2),
        avg_CO2=round(cities['Edmonton']['avg_CO2'],2),
        avg_PM25=round(cities['Edmonton']['avg_PM25'],2))

@app.route("/adrar")
def adrar():
    return render_template("mapplot.html",
        dates=cities['Adrar']['dates'],
        poll="PM25",
        min_lon=cities["Adrar"]['min_lon'],
        max_lon=cities['Adrar']['max_lon'],
        min_lat=cities["Adrar"]['min_lat'],
        max_lat=cities['Adrar']['max_lat'],
        mean_lat=cities['Adrar']['avg_lat'],
        mean_lon=cities['Adrar']['avg_lon'],
        avg_NO=round(cities['Adrar']['avg_NO'],2),
        avg_CO=round(cities['Adrar']['avg_CO'],2),
        avg_CO2=round(cities['Adrar']['avg_CO2'],2),
        avg_PM25=round(cities['Adrar']['avg_PM25'],2))

@app.route("/rice")
def rice():
    return render_template("mapplot.html",
        dates=cities['Rice County']['dates'],
        poll="PM25",
        min_lon=cities["Rice County"]['min_lon'],
        max_lon=cities['Rice County']['max_lon'],
        min_lat=cities["Rice County"]['min_lat'],
        max_lat=cities['Rice County']['max_lat'],
        mean_lat=cities['Rice County']['avg_lat'],
        mean_lon=cities['Rice County']['avg_lon'],
        avg_NO=round(cities['Rice County']['avg_NO'],2),
        avg_CO=round(cities['Rice County']['avg_CO'],2),
        avg_CO2=round(cities['Rice County']['avg_CO2'],2),
        avg_PM25=round(cities['Rice County']['avg_PM25'],2))

@app.route("/brisbane")
def brisbane():
    return render_template("mapplot.html",
        dates=cities['Brisbane']['dates'],
        poll="PM25",
        min_lon=cities["Brisbane"]['min_lon'],
        max_lon=cities['Brisbane']['max_lon'],
        min_lat=cities["Brisbane"]['min_lat'],
        max_lat=cities['Brisbane']['max_lat'],
        mean_lat=cities['Brisbane']['avg_lat'],
        mean_lon=cities['Brisbane']['avg_lon'],
        avg_NO=round(cities['Brisbane']['avg_NO'],2),
        avg_CO=round(cities['Brisbane']['avg_CO'],2),
        avg_CO2=round(cities['Brisbane']['avg_CO2'],2),
        avg_PM25=round(cities['Brisbane']['avg_PM25'],2))

@app.route("/latrobe")
def latrobe():
    return render_template("mapplot.html",
        dates=cities['Latrobe City']['dates'],
        poll="PM25",
        min_lon=cities["Latrobe City"]['min_lon'],
        max_lon=cities['Latrobe City']['max_lon'],
        min_lat=cities["Latrobe City"]['min_lat'],
        max_lat=cities['Latrobe City']['max_lat'],
        mean_lat=cities['Latrobe City']['avg_lat'],
        mean_lon=cities['Latrobe City']['avg_lon'],
        avg_NO=round(cities['Latrobe City']['avg_NO'],2),
        avg_CO=round(cities['Latrobe City']['avg_CO'],2),
        avg_CO2=round(cities['Latrobe City']['avg_CO2'],2),
        avg_PM25=round(cities['Latrobe City']['avg_PM25'],2))

@app.route("/bucuresti")
def bucuresti():
    return render_template("mapplot.html",
        dates=cities['Bucuresti']['dates'],
        poll="PM25",
        min_lon=cities["Bucuresti"]['min_lon'],
        max_lon=cities['Bucuresti']['max_lon'],
        min_lat=cities["Bucuresti"]['min_lat'],
        max_lat=cities['Bucuresti']['max_lat'],
        mean_lat=cities['Bucuresti']['avg_lat'],
        mean_lon=cities['Bucuresti']['avg_lon'],
        avg_NO=round(cities['Bucuresti']['avg_NO'],2),
        avg_CO=round(cities['Bucuresti']['avg_CO'],2),
        avg_CO2=round(cities['Bucuresti']['avg_CO2'],2),
        avg_PM25=round(cities['Bucuresti']['avg_PM25'],2))

@app.route("/hamilton")
def hamilton():
    return render_template("mapplot.html",
        dates=cities['Hamilton County']['dates'],
        poll="PM25",
        min_lon=cities["Hamilton County"]['min_lon'],
        max_lon=cities['Hamilton County']['max_lon'],
        min_lat=cities["Hamilton County"]['min_lat'],
        max_lat=cities['Hamilton County']['max_lat'],
        mean_lat=cities['Hamilton County']['avg_lat'],
        mean_lon=cities['Hamilton County']['avg_lon'],
        avg_NO=round(cities['Hamilton County']['avg_NO'],2),
        avg_CO=round(cities['Hamilton County']['avg_CO'],2),
        avg_CO2=round(cities['Hamilton County']['avg_CO2'],2),
        avg_PM25=round(cities['Hamilton County']['avg_PM25'],2))

@app.route("/qld")
def qld():
    return render_template("mapplot.html",
        dates=cities['QLD']['dates'],
        poll="PM25",
        min_lon=cities["QLD"]['min_lon'],
        max_lon=cities['QLD']['max_lon'],
        min_lat=cities["QLD"]['min_lat'],
        max_lat=cities['QLD']['max_lat'],
        mean_lat=cities['QLD']['avg_lat'],
        mean_lon=cities['QLD']['avg_lon'],
        avg_NO=round(cities['QLD']['avg_NO'],2),
        avg_CO=round(cities['QLD']['avg_CO'],2),
        avg_CO2=round(cities['QLD']['avg_CO2'],2),
        avg_PM25=round(cities['QLD']['avg_PM25'],2))

@app.route("/philly")
def philly():
    return render_template("mapplot.html",
        dates=cities['Philadelphia County']['dates'],
        poll="PM25",
        min_lon=cities["Philadelphia County"]['min_lon'],
        max_lon=cities['Philadelphia County']['max_lon'],
        min_lat=cities["Philadelphia County"]['min_lat'],
        max_lat=cities['Philadelphia County']['max_lat'],
        mean_lat=cities['Philadelphia County']['avg_lat'],
        mean_lon=cities['Philadelphia County']['avg_lon'],
        avg_NO=round(cities['Philadelphia County']['avg_NO'],2),
        avg_CO=round(cities['Philadelphia County']['avg_CO'],2),
        avg_CO2=round(cities['Philadelphia County']['avg_CO2'],2),
        avg_PM25=round(cities['Philadelphia County']['avg_PM25'],2))

if __name__ == '__main__':
    event = pickle.load(open('event.pkl','rb'))

    app.run(host='0.0.0.0', port = 8000, debug = False)
