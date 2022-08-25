from turtle import st
from click import pass_context
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
# Crear una carpeta oculta, con un archivo de registro


def __create_hidden_folder():
    """Crear una carpeta oculta en la ubicación donde se esta ejecutando la función"""

    os.mkdir('makesens_data')
    os.system('attrib +h makesens_data')

    with open('makesens_data/registro.json', 'w') as fp:
        json.dump({}, fp)

# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------

# Guardar el registro de cada descarga


def __save_register(id_device: str, start_date: int, end_date: int, sample_rate: str, file_name: str):
    with open('makesens_data/registro.json', 'r') as fp:
        registro = json.load(fp)

    if id_device in registro.keys():
        registro[id_device].append(
            [sample_rate, start_date, end_date, file_name])
    else:
        registro.setdefault(
            id_device, [[sample_rate, start_date, end_date, file_name]])
    with open('makesens_data/registro.json', 'w') as fp:
        json.dump(registro, fp)

# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------

# Guardar los datos en el respaldo, con su respectivo registro


def __save_in_backup(data, id_device: str, start_date: int, end_date: int, sample_rate: str) -> str:
    archivos = os.listdir()

    file_name = id_device + '_' + \
        str(start_date) + '_' + str(end_date) + '_' + sample_rate + '.csv'
    if 'makesens_data' in archivos:
        data.to_csv('makesens_data/' + file_name)
        __save_register(id_device, start_date, end_date,
                        sample_rate, file_name)
    else:
        __create_hidden_folder()
        data.to_csv('makesens_data/' + file_name)
        __save_register(id_device, start_date, end_date,
                        sample_rate, file_name)

    return file_name

# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------

# Descargar los datos desde la API


def __download(id_device: str, start_date: int, end_date: int, sample_rate: str, token: str):
    """Descarga los datos del servidor de makesens en base a peticiones http"""

    data = []
    data_ = pd.DataFrame()
    tmin: int = start_date
    while tmin < end_date:
        url = 'https://makesens.aws.thinger.io/v1/users/MakeSens/buckets/B' + id_device + '/data?agg=1' + sample_rate + \
            '&agg_type=mean&items=1000&max_ts=' + \
            str(end_date) + '000&min_ts=' + str(tmin) + \
            '000&sort=asc&authorization=' + token
        d = json.loads(requests.get(url).content)

        try:
            if tmin == (d[-1]['ts']//1000) + 1:
                break
            data += d
            tmin = (d[-1]['ts']//1000) + 1
            data_ = pd.DataFrame([i['mean'] for i in data], index=[datetime.utcfromtimestamp(
                i['ts']/1000).strftime('%Y-%m-%d %H:%M:%S') for i in data])
        except IndexError:
            break

    file_name = __save_in_backup(
        data_, id_device, start_date, end_date, sample_rate)
    return file_name

# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------

# Identificar que datos hay en backup y cuales faltan


def __in_backup(id_device: str, start_date: str, end_date: str, sample_rate: str):
    missing: List[Tuple(int, int)] = []
    file_names: List[str] = []

    with open('makesens_data/registro.json', 'r') as fp:
        registro = json.load(fp)

    if id_device in registro.keys():
        registro[id_device] = sorted(
            registro[id_device], key=lambda rango: rango[1])
        for i in range(0, len(registro[id_device])):
            if registro[id_device][i][0] == sample_rate:
                if (start_date < registro[id_device][i][1]) and (end_date < registro[id_device][i][1]):
                    missing.append((start_date, end_date))
                    #print('a')
                    if (start_date < registro[id_device][i][2]) and (end_date < registro[id_device][i][2]):
                        break
                    elif (start_date < registro[id_device][i][2]) and (end_date > registro[id_device][i][2]):
                        start_date = registro[id_device][i][2]
                elif (start_date < registro[id_device][i][1]) and (end_date > registro[id_device][i][1]):
                    #print('b')
                    missing.append((start_date, registro[id_device][i][1]))
                    start_date = registro[id_device][i][1]
                    if (start_date < registro[id_device][i][2]) and (end_date <= registro[id_device][i][2]):
                        file_names.append(registro[id_device][i][3])
                        break
                    elif (start_date < registro[id_device][i][2]) and (end_date > registro[id_device][i][2]):
                        file_names.append(registro[id_device][i][3])
                        start_date = registro[id_device][i][2]
                elif (start_date >= registro[id_device][i][1]) and (start_date < registro[id_device][i][2]):
                    file_names.append(registro[id_device][i][3])
                    start_date = registro[id_device][i][2]
                if (i == len(registro[id_device])-1) and (start_date >= registro[id_device][i][2]):
                    if (start_date == end_date) or (end_date <= registro[id_device][i][2]):
                        break
                    else:
                        #print('c')
                        missing.append((start_date, end_date))
                
            else:
                if i == (len(registro[id_device])-1):
                    #print('d')
                    missing.append((start_date, end_date)) 
                else:
                    pass
    else:
        missing.append((start_date, end_date))
        #print('e')
    
    #print(file_names)
    return missing, file_names
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
#Luego de entregar unos datos se debe actualizar el registro

def __cutdata(data, start, end):

    """
    Parameters:
        data -> data to cut
        start:str -> index of startup
        end:str -> end index
    
    Returns:
        data cut out    
    """
    mask = (data.index >= start) & (data.index <= end)
    data = data[mask]
    return data  

# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------

def download_data(id_device: str, start_date: str, end_date: str, sample_rate: str, token: str, format:str = None):
    
    start = int((datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S") - datetime(1970, 1, 1)).total_seconds())
    end = int((datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S") -  datetime(1970, 1, 1)).total_seconds())

    file_names = []
    if 'makesens_data' in os.listdir():
        missing, file_names = __in_backup(id_device, start, end, sample_rate)
        #print(missing)
        if len(missing) == 0:
            pass
        else:
            for range in missing:
                name = __download(id_device,range[0],range[1],sample_rate,token)
                file_names.append(name)
        
    else:
        name = __download(id_device, start, end, sample_rate, token)
        file_names.append(name)

    data = pd.DataFrame()
    for i in file_names:
        dat = pd.read_csv('makesens_data/'+i)
        dat.index = dat.iloc[:,0]
        dat = dat.drop([dat.columns[0]],axis=1)
        data = pd.concat([data,dat],axis=0)
        
    data = data.sort_index()
    data = data.drop_duplicates()
    if len(data) != 0:
        new_start = int((datetime.strptime(data.index[0], "%Y-%m-%d %H:%M:%S") - datetime(1970, 1, 1)).total_seconds())
        new_end = int((datetime.strptime(data.index[-1], "%Y-%m-%d %H:%M:%S") -  datetime(1970, 1, 1)).total_seconds())
    #print(data.index[0],data.index[-1])
    if len(file_names) == 1:
        pass
    else:
        for i in file_names:
            os.remove('makesens_data/'+ i)

        with open('makesens_data/registro.json', 'r') as fp:
            registro = json.load(fp)


        index = [registro[id_device].index(i) for i in registro[id_device] if i[3] in file_names]

        for i in sorted(index,reverse=True):
              del registro[id_device][i]
            
        with open('makesens_data/registro.json', 'w') as fp:
            json.dump(registro, fp)
         
        if len(data) != 0:     
            __save_in_backup(data, id_device, new_start, new_end, sample_rate)
    
    start_ = start_date.replace(':','_') 
    end_ = end_date.replace(':','_')
    
    if format == None:
        pass
    elif format == 'csv':
        data.to_csv(id_device + '_'+ start_  +'_' + end_ + '_ ' + sample_rate  +'.csv')
    elif format == 'xlsx':
        data.to_excel(id_device + '_'+ start_  +'_' + end_ + '_ ' + sample_rate  +'.xlsx')
    else:
        print('El formato no es valido. Formatos validos: csv y xlsx')
        
    data_ = data
    data_ = __cutdata(data,start_date,end_date)
    return data_

# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
def __gradient_plot(data,scale,y_label,sample_rate):
    
    data.index = pd.DatetimeIndex(data.index)
    a = pd.date_range(data.index[0],data.index[-1], freq='h')
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
    
    points = np.array([x, y]).T.reshape(-1, 1, 2)
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

def gradient_pm10(id_device:str,start_date:str,end_date:str,sample_rate:str, token:str):
    data = download_data(id_device,start_date,end_date,sample_rate, token,None)
    
    data.index
    __gradient_plot(data.pm10_1,(54,255),'PM10 ', sample_rate)

def gradient_pm2_5(id_device:str,start_date:str,end_date:str,sample_rate:str, token:str):
    data = download_data(id_device,start_date,end_date,sample_rate, token,None)
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
    

def heatmap_pm10(id_device:str,start_date:str,end_date:str,token:str):
    data = download_data(id_device,start_date,end_date,'h', token,None)
    data.index = pd.DatetimeIndex(data.index)
    data = data.pm10_1
    
    _heatmap_plot(data,(54,255),'PM10')
    
def heatmap_pm2_5(id_device:str,start_date:str,end_date:str, token:str):
    data = download_data(id_device,start_date,end_date,'h', token,None)
    data.index = pd.DatetimeIndex(data.index)
    data = data.pm25_1
    
    _heatmap_plot(data,(12,251),'PM2.5')
    
