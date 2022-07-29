from turtle import st
from click import pass_context
import pandas as pd
import requests
import json
from datetime import datetime
import os
from typing import List, Tuple
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
            if (start_date < registro[id_device][i][1]) and (end_date < registro[id_device][i][1]):
                missing.append((start_date, end_date))
                if (start_date < registro[id_device][i][2]) and (end_date < registro[id_device][i][2]):
                    break
                elif (start_date < registro[id_device][i][2]) and (end_date > registro[id_device][i][2]):
                    start_date = registro[id_device][i][2]
            elif (start_date < registro[id_device][i][1]) and (end_date > registro[id_device][i][1]):
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
                if start_date == end_date:
                    break
                missing.append((start_date, end_date))
    else:
        missing.append((start_date, end_date))

    return missing, file_names
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
#Luego de entregar unos datos se debe actualizar el registro

    

# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------


def download_data(id_device: str, start_date: str, end_date: str, sample_rate: str, token: str, format:str):
    
    start = int((datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S") - datetime(1970, 1, 1)).total_seconds())
    end = int((datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S") -  datetime(1970, 1, 1)).total_seconds())

    file_names = []
    if 'makesens_data' in os.listdir():
        missing, file_names = __in_backup(id_device, start, end, sample_rate)
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
        
        __save_in_backup(data, id_device, start, end, sample_rate)
    
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
        
    return data

