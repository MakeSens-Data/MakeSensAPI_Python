import pandas as pd
import requests
import json
from datetime import datetime

def download_data(IdDevice:str,start:int,end:int,frecuency:str,token:str):   
    start = int((datetime.strptime(start,"%Y-%m-%d %H:%M:%S") - datetime(1970, 1, 1)).total_seconds())
    end = int((datetime.strptime(end,"%Y-%m-%d %H:%M:%S") - datetime(1970, 1, 1)).total_seconds())
    data = []
    data_ = pd.DataFrame()
    tmin  = start
    while tmin < end :
        print(tmin)
        url = 'https://makesens.aws.thinger.io/v1/users/MakeSens/buckets/B' + IdDevice + '/data?agg=1'+frecuency+'&agg_type=mean&items=1000&max_ts=' + str(end) + '000&min_ts='+ str(tmin) +'000&sort=asc&authorization=' + token
        d = json.loads(requests.get(url).content)
        if tmin == (d[-1]['ts']//1000) + 1:
            break
        data+=d
        tmin=(d[-1]['ts']//1000) + 1
        data_ = pd.DataFrame([i['mean'] for i in data],index=[datetime.utcfromtimestamp(i['ts']/1000).strftime('%Y-%m-%d %H:%M:%S') for i in data])
    return (data_)

def save_data(IdDevice:str,start:int,end:int,frecuency:str,token:str,format_:str):
    data = download_data(IdDevice,start,end,frecuency,token)
    if format_ == 'csv':
        data.to_csv(IdDevice + frecuency  +'.csv')
    if format_ == 'xlsx':
        data.to_excel(IdDevice + frecuency  +'.xlsx')