import pandas as pd
import requests
import json
from datetime import datetime

def download_data(IdDevice:str,start:int,end:int,frecuency:str,token:str,format_:str = None):   
    start_ = int((datetime.strptime(start,"%Y-%m-%d %H:%M:%S") - datetime(1970, 1, 1)).total_seconds())
    end_ = int((datetime.strptime(end,"%Y-%m-%d %H:%M:%S") - datetime(1970, 1, 1)).total_seconds())
    data = []
    data_ = pd.DataFrame()
    tmin  = start_
    while tmin < end_ :
        url = 'https://makesens.aws.thinger.io/v1/users/MakeSens/buckets/B' + IdDevice + '/data?agg=1'+frecuency+'&agg_type=mean&items=1000&max_ts=' + str(end_) + '000&min_ts='+ str(tmin) +'000&sort=asc&authorization=' + token
        d = json.loads(requests.get(url).content)
        if tmin == (d[-1]['ts']//1000) + 1:
            break
        data+=d
        tmin=(d[-1]['ts']//1000) + 1
        data_ = pd.DataFrame([i['mean'] for i in data],index=[datetime.utcfromtimestamp(i['ts']/1000).strftime('%Y-%m-%d %H:%M:%S') for i in data])
    

    start = start.replace(':','-')
    end = end.replace(':','-')
    if format_ == None:
        pass
    elif format_ == 'csv':
        data_.to_csv(IdDevice + '_'+ start  +'_' + end + '_ ' + frecuency  +'.csv')
    elif format_ == 'xlsx':
        data_.to_excel(IdDevice + '_'+ start  +'_' + end + '_ ' + frecuency  +'.xlsx')
    else:
        print('El formato no es valido. Formatos validos: csv y xlsx')
    return (data_)