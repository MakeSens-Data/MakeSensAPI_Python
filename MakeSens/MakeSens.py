import pandas as pd
import requests
import json
from datetime import datetime
import os
from typing import List, Tuple
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def __save_data(data,name:str,format:str)-> None:
    if format == 'csv':
        data.to_csv(name + '.csv')
    elif format == 'xlsx':
        data.to_excel(name +'.xlsx')
    else:
        print('Formato no valido. Formatos validos: csv y xlsx')


def download_data(id_device:str,start_date:str,end_date:str, sample_rate:str = None,format:str = None, fields:str = None):
    
    start:int = int((datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S") - datetime(1970, 1, 1)).total_seconds())
    end:int = int((datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S") -  datetime(1970, 1, 1)).total_seconds())
    
    dat:list = []
    tmin:int = start
        
    while tmin < end:
        if sample_rate != None:
            if fields == None:
                url = f'https://api.makesens.co/ambiental/metricas/{id_device}/data?agg=1{sample_rate}&agg_type=mean&items=1000&max_ts={str(end * 1000)}&min_ts={str(tmin * 1000)}'
            else:
                url =  f'https://api.makesens.co/ambiental/metricas/{id_device}/data?agg=1{sample_rate}&agg_type=mean&fields={fields}&items=1000&max_ts={str(end * 1000)}&min_ts={str(tmin * 1000)}'
        else:
            if fields == None:
                url = f'https://api.makesens.co/ambiental/metricas/{id_device}/data?items=1000&max_ts={str(end * 1000)}&min_ts={str(tmin * 1000)}'
            else:
                url =  f'https://api.makesens.co/ambiental/metricas/{id_device}/data?fields={fields}&items=1000&max_ts={str(end * 1000)}&min_ts={str(tmin * 1000)}'

        rta = requests.get(url).content
        d = json.loads(rta)
        try:
            if tmin == (d[-1]['ts']//1000) + 1:
                break
            dat = dat + d
            tmin = (d[-1]['ts']//1000) + 1
        except IndexError:
            break
       
    data = pd.DataFrame([i['val'] for i in dat], index=[datetime.utcfromtimestamp(i['ts']/1000).strftime('%Y-%m-%d %H:%M:%S') for i in dat])
    
    start_ = start_date.replace(':','_') 
    end_ = end_date.replace(':','_')
    if sample_rate == None:
        name = id_device + '_'+ start_  +'_' + end_ 
    else:
        name = id_device + '_'+ start_  +'_' + end_ + '_ ' + sample_rate
    
    if format != None:
        __save_data(data,name,format)    
    
        
    return data
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
def __gradient_plot(data,scale,y_label,sample_rate):
    
    if sample_rate == 'm':
        sample_rate = 'T'
    elif sample_rate == 'w':
        sample_rate ='7d'
    
    data.index = pd.DatetimeIndex(data.index)
    a = pd.date_range(data.index[0],data.index[-1], freq=sample_rate)
    s = []
    for i in a:
        if i in data.index:
            s.append(data[i])
        else: 
            s.append(np.nan)

    dat = pd.DataFrame(index=a)
    dat['PM'] = s

    dat.index = dat.index.strftime("%Y-%m-%d %H:%M:%S")     
    
    colorlist=["green", "yellow",'Orange', "red", 'Purple','Brown']
    newcmp = LinearSegmentedColormap.from_list('testCmap', colors=colorlist, N=256)
    
    y_ = np.array(list(dat['PM']))
    x_ = np.linspace(1, len(y_),len(y_))#3552, 3552)
    
    x = np.linspace(1, len(y_), 10000)
    y = np.interp(x, x_, y_)
    
    points = np.array([x-1, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, ax = plt.subplots(figsize=(15, 5))

    norm = plt.Normalize(scale[0], scale[1])
    lc = LineCollection(segments, cmap=newcmp, norm=norm)

    lc.set_array(y)
    lc.set_linewidth(1)
    line = ax.add_collection(lc)
    dat['PM'].plot(lw=0)
    plt.colorbar(line, ax=ax)
    ax.set_ylim(min(y)-10, max(y)+ 10)
    plt.ylabel(y_label+'$\mu g / m^3$', fontsize = 14)
    plt.xlabel('Estampa temporal', fontsize = 14)
    plt.gcf().autofmt_xdate()
    plt.show()

def gradient_pm10(id_device:str,start_date:str,end_date:str,sample_rate:str):
    data = download_data(id_device,start_date,end_date,sample_rate, fields='pm10_1')
    data['ts'] = data.index
    data = data.drop_duplicates(subset = ['ts'])
    
    __gradient_plot(data.pm10_1,(54,255),'PM10 ', sample_rate)

def gradient_pm2_5(id_device:str,start_date:str,end_date:str,sample_rate:str):
    data = download_data(id_device,start_date,end_date,sample_rate, fields='pm25_1')
    data['ts'] = data.index
    data = data.drop_duplicates(subset = ['ts'])
    __gradient_plot(data.pm25_1,(12,251), 'PM2.5 ',sample_rate)

# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------

def _heatmap_plot (data,scale,title):
    colorlist=["green", "yellow",'Orange', "red", 'Purple','Brown']
    newcmp = LinearSegmentedColormap.from_list('testCmap', colors=colorlist, N=256)
    norm = plt.Normalize(scale[0], scale[1])
    
    date = pd.date_range(data.index.date[0],data.index.date[-1]).date
    hours = range(0,24)
    
    mapa = pd.DataFrame(columns=date, index=hours,dtype="float")
    
    for i in range(0,len(date)):
        dat = data[data.index.date == date[i]]
        for j in range(0,len(dat)):
            fila = dat.index.hour[j]
            mapa[date[i]][fila] = dat[j]
    
    plt.figure(figsize=(10,8))
    ax = sns.heatmap(mapa, cmap=newcmp,norm=norm)
    plt.ylabel('Horas', fontsize = 16)
    plt.xlabel('Estampa temporal', fontsize = 16)
    plt.title(title + '$\mu g / m^3$', fontsize = 16)
    plt.show()
    

def heatmap_pm10(id_device:str,start_date:str,end_date:str):
    data = download_data(id_device,start_date,end_date,'h',fields='pm10_1')
    data.index = pd.DatetimeIndex(data.index)
    data = data.pm10_1
    
    _heatmap_plot(data,(54,255),'PM10')
    
def heatmap_pm2_5(id_device:str,start_date:str,end_date:str):
    data = download_data(id_device,start_date,end_date,'h', fields='pm25_1')
    data.index = pd.DatetimeIndex(data.index)
    data = data.pm25_1
    
    _heatmap_plot(data,(12,251),'PM2.5')


# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------

def weekly_profile (id_device,start_date,end_date,field):
    fields = {'PM10':{'variable':'pm10_1','unidades':'[$\mu g/m^3$ ]'},'PM2.5':{'variable':'pm25_1','unidades':'[$\mu g/m^3$ ]'}
             ,'CO2':{'ppm'}}
    
    var = fields[field]['variable']
    unidad = fields[field]['unidades']
    
    data = download_data(id_device, start_date,end_date,'h',fields = var)
    data.index = pd.DatetimeIndex(data.index)
    days = range(0,7)
    hours = range(0,24)
    
    data['day'] = [i.weekday() for i in data.index]
    data['hour'] = [i.hour for i in data.index]
    variable_mean = []
    variable_std = []
    for day in days:
        for hour in hours:
            variable = data[(data.day == day) & (data.hour == hour)][var]
            variable_mean.append(variable.mean())
            variable_std.append(variable.std())
    
    a = min(np.array(variable_mean) - np.array(variable_std))
    b = max(np.array(variable_mean) + np.array(variable_std))
    
    x = [i for i in range(168)]
    plt.figure(figsize = (18,4))
    plt.plot(x,np.array(variable_mean))
    plt.fill_between(x,np.array(variable_mean) - np.array(variable_std),np.array(variable_mean) + np.array(variable_std), color = 'r', alpha = 0.2)
    plt.xticks(np.linspace(0,162,28), ['0','6','12','18'] * 7  )

    plt.hlines(b + 5,0,167,color = 'k')
    for i in np.linspace(0,168,8)[1:-1]:
        plt.vlines(i,a,b+15, color = 'k', ls = '--',lw = 1)
    
    name_days = ['Lunes','Martes', 'Miercoles','Jueves','Viernes','Sabado','Domingo']
    position_x = [9,30,55,80,103,128,150]
    for i in range(0,len(name_days)):
        plt.text(position_x[i],b+8,name_days[i],fontsize = 13)

    plt.xlim(0,167)
    plt.ylim(a,b+15)

    plt.xlabel('Horas',fontsize = 14)
    plt.ylabel(f'{field} {unidad}',fontsize = 14)
    plt.show()   


if __name__ == '__main__':
    data = download_data('mE1_00004','2023-03-14 00:00:00', '2023-03-26 00:00:00')
    print(data)