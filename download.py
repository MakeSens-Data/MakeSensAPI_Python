import numpy as np 
import pandas as pd
from datetime import datetime

from pandas.io.json import json_normalize
import requests
from requests.auth import HTTPBasicAuth
import json


def download_eva(IdDevice:str,tmin:int,tmax:int,frecuency:str, format_:str='None'):
    """
    Parameters:
        bucket:str -> station identifier
        tmin:int -> maximum time in unix format
        tmax:int -> minimum time in unix format
        frecuency:str -> sampling frequency
                         h -> hour
                         m -> minutes
        format_:str = data format (.csv, .xlxs)                      
    """
    s = requests.Session()
    data_ = []
    while tmin < tmax :
        url_ = 'https://makesens.aws.thinger.io/v1/users/MakeSens/buckets/B' + IdDevice + '/data?agg=1'+frecuency+'&agg_type=mean&items=1000&max_ts=' + str(tmax) + '000&min_ts='+ str(tmin) +'000&sort=asc&authorization=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJUT0tFTl9JQ1AiLCJ1c3IiOiJNYWtlU2VucyJ9.aFSgJgp6YRhkbu0jGmkkRuwVn07RR-zxEYlFHwrVJVc'
        try:
            data = s.get(url_)
            data_ = data_ + data.json()
            try:
                if tmin == data.json()[-1]['ts']//1000: 
                    break
                else:
                    tmin = data.json()[-1]['ts']//1000
                dict_ = {'ts': [], 'humedad' : [], 'irradiancia' : [], 'pm10_1':[],'pm10_2': [], 'pm25_1': [],'pm25_2': [],'precipitacion': [],
             'presion': [],'temperatura' : []}

                for i in range(0,len(data_)):
                    dict_['ts'].append(datetime.fromtimestamp(data_[i]['ts']/1000).strftime('%Y-%m-%d %H:%M:%S'))
                    for j in list(dict_.keys())[1:]:
                        if  j in list(data_[i]['mean'].keys()):
                            dict_[j].append(data_[i]['mean'][j])
                        else:
                            dict_[j].append(np.nan)
                
            except IndexError:
                print('No hay datos en este intervalo')
                break
        except TypeError:
            print('the requested bucket does not exist')
            break
    
    _data = pd.DataFrame(data=dict_)
    if format_ == 'None':
        return _data
    else:        
        if format_ == 'csv':      
            _data.to_csv(IdDevice + '_' + str(tmin) + '_to_' + str(tmax) + '_' + frecuency  +'.csv')
        elif format_ == 'xlsx':
            _data.to_excel(IdDevice + '_' + str(tmin) + '_to_' + str(tmax) + '_' + frecuency  +'.xlsx')