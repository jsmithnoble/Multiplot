import os

def export_for_visit(df, city, poll, date):
    '''
    INPUT
    df: A data frame already subset by city
    city: string city names
    poll: string type name of pollutant to be plotted
    date: string yyyy-mm-dd
    Creates a data file located at data/city/poll_date.3D
    '''
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

def format_visit_plots():
    '''
    Iterate through visit plots in f_path1 and save formatted plots to f_path2
    '''
    f_path1 = '/Users/jakenoble/DSI/multiplot/plots/visIt'
    f_path2 = '/Users/jakenoble/DSI/multiplot/images'
    n = 1
    for city in cities:
        for poll in ['PM25']:#,'CO','CO2','NO']:
            for date in cities[city]['dates']:
                call(['open','{}/{}/{}/{}_{}_0000.png'.format(f_path1,city.replace(' ','_'),poll,poll,date)])
                if not os.path.exists('{}/{}/{}'.format(f_path2,city.replace(' ','_'),poll)):
                    os.makedirs('{}/{}/{}'.format(f_path2,city.replace(' ','_'),poll))
                call(['screencapture','-T','1','-R','175,121,645,623', '{}/{}/{}/{}_{}.png'.format(f_path2,city.replace(' ','_'),poll,poll,date)])
                call(['osascript', '-e', 'quit app "Preview"'])
                print "\n made %i plots" %n
                print '{}/{}/{}/{}_{}.png'.format(f_path2,city,poll,poll,date)
                n+=1
