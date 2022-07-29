import pandas as pd
import requests
import json
from datetime import datetime
import os

def __download(IdDevice:str,start,end,frecuency:str,token:str):
    if type(start) == str:
        start_ = int((datetime.strptime(start,"%Y-%m-%d %H:%M:%S") - datetime(1970, 1, 1)).total_seconds())
        end_ = int((datetime.strptime(end,"%Y-%m-%d %H:%M:%S") - datetime(1970, 1, 1)).total_seconds())

        start = start.replace(':','-')
        end = end.replace(':','-')
    else: 
        start_  = start
        end_ = end

        start = str(start)
        end = str(end)
    data = []
    data_ = pd.DataFrame()
    tmin  = start_
    while tmin < end_ :
        url = 'https://makesens.aws.thinger.io/v1/users/MakeSens/buckets/B' + IdDevice + '/data?agg=1'+frecuency+'&agg_type=mean&items=1000&max_ts=' + str(end_) + '000&min_ts='+ str(tmin) +'000&sort=asc&authorization=' + token
        d = json.loads(requests.get(url).content)
        try:
            if tmin == (d[-1]['ts']//1000) + 1:
                break
            data+=d
            tmin=(d[-1]['ts']//1000) + 1
            data_ = pd.DataFrame([i['mean'] for i in data],index=[datetime.utcfromtimestamp(i['ts']/1000).strftime('%Y-%m-%d %H:%M:%S') for i in data])
        except IndexError:
            break

    return (data_, start_,end_) 
    
#Función para verificar si la carpeta oculta ya esta creada y sino crearla
def __crearCarpeta(nombre:str):
    archivos = os.listdir()

    if nombre in archivos:
        return False
    elif nombre not in archivos:
        os.mkdir(nombre)
        os.system('attrib +h ' + nombre) 
        
        with open('carpetaOculta/registro.json', 'w') as fp:
            json.dump({}, fp)
        return True

#Verificar si hay datos utiles ya descargados
def __inbackup (IdDevice,start,end,frecuency):
    start_ = int((datetime.strptime(start,"%Y-%m-%d %H:%M:%S") - datetime(1970, 1, 1)).total_seconds())
    end_ = int((datetime.strptime(end,"%Y-%m-%d %H:%M:%S") - datetime(1970, 1, 1)).total_seconds())

    with open('carpetaOculta/registro.json', 'r') as fp:
        registro = json.load(fp)

    resultado = []
    datosEn = []
    if IdDevice in registro.keys(): #Verificar si esta el  dispositivo
        for i in range(0,len(registro[IdDevice])):
            if frecuency == registro[IdDevice][i][0]: # Modificar lo de la frecuencia
                if start_ >= (registro[IdDevice][i][2]) or (end_ <= registro[IdDevice][i][1]):
                    resultado.append(False)
                else: 
                    resultado.append(True)
                    datosEn.append(i)

    result = True in resultado

    return result,datosEn

#Cargar los datos que son utiles
def __loadBackup(IdDevice,start,end,datosEn):
    start_ = int((datetime.strptime(start,"%Y-%m-%d %H:%M:%S") - datetime(1970, 1, 1)).total_seconds())
    end_ = int((datetime.strptime(end,"%Y-%m-%d %H:%M:%S") - datetime(1970, 1, 1)).total_seconds())

    faltan = []
    datas = []

    with open('carpetaOculta/registro.json', 'r') as fp:
        registro = json.load(fp)

    for i in range(0,len(datosEn)):
        if start_ < registro[IdDevice][datosEn[i]][1]: #Faltan datos al inicio
            faltan.append((start_,registro[IdDevice][datosEn[i]][1])) 
            if end_ > registro[IdDevice][datosEn[i]][2]: #Faltan datos al final
                faltan.append((registro[IdDevice][datosEn[i]][2],end_))
                    
        else: #Se inicia con los datos que ya estan
            if end_ > registro[IdDevice][datosEn[i]][2]: #Faltan datos al final
                faltan.append((registro[IdDevice][datosEn[i]][2],end_))

        data_backup = pd.read_csv(registro[IdDevice][datosEn[i]][3])
        data_backup.index = data_backup.iloc[:,0]
        data_backup = data_backup.drop([data_backup.columns[0]],axis=1)
        datas.append(data_backup)
    
    if len(datas) == 1:
        datt = datas[0]
    else:
        for i in range(0,len(datas)-1):
            if i == 0:
                datt = pd.concat([datas[i],datas[i+1]])
            else:
                datt = pd.concat([datt,datas[i+1]])
    return faltan,datt,[registro[IdDevice][i][3] for i in datosEn]


#Entrega los datos al usuario
def download_data (IdDevice,start,end,frecuency,token,format_:str = None):   
    a = __crearCarpeta('carpetaOculta') #Crear carpeta oculta

    #Verificar si ya hay archivos
    if not a :
        backup, datosEn = __inbackup(IdDevice,start,end,frecuency)
    if a: 
        backup = False
    
    #Cargar o descargar los datos
    registro = {}
    if backup == False:
        data, start_ , end_ = __download(IdDevice,start,end,frecuency,token)
        nombre_archivo = 'carpetaOculta/' + IdDevice + '_' + str(start_) + '_'+ str(end_)+ '_' + frecuency + '.csv'
        data.to_csv(nombre_archivo)

        with open('carpetaOculta/registro.json', 'r') as fp:
            registro = json.load(fp)
        
        if IdDevice in registro.keys():
            registro[IdDevice].append([frecuency,start_,end_,nombre_archivo])
        else:
            registro.setdefault(IdDevice,[[frecuency,start_,end_,nombre_archivo]])
        
        with open('carpetaOculta/registro.json', 'w') as fp:
            json.dump(registro, fp)

    elif backup == True:
        faltan , data_backup, nombres_backup = __loadBackup(IdDevice,start,end,datosEn)
        data = data_backup

        for i in faltan:
            data1, start_ , end_ = __download(IdDevice,i[0],i[1],frecuency,token)
            data = pd.concat([data,data1],axis = 0)

        data = data.sort_index()
        data = data.drop_duplicates()

        tmin = int((datetime.strptime(data.index[0],"%Y-%m-%d %H:%M:%S") - datetime(1970, 1, 1)).total_seconds())
        tmax = int((datetime.strptime(data.index[-1],"%Y-%m-%d %H:%M:%S") - datetime(1970, 1, 1)).total_seconds())
        
        if len(faltan) != 0:
            nombre_archivo = 'carpetaOculta/' + IdDevice + '_' + str(tmin) + '_'+ str(tmax)+ '_' + frecuency + '.csv'

            with open('carpetaOculta/registro.json', 'r') as fp:
                registro = json.load(fp)

    
            for i in sorted(datosEn,reverse=True):
                del registro[IdDevice][i]

            registro[IdDevice].append([frecuency,tmin,tmax,nombre_archivo])
            
            with open('carpetaOculta/registro.json', 'w') as fp:
                json.dump(registro, fp)
            
            for i in nombres_backup:
                os.remove(i)

            data.to_csv(nombre_archivo)  

    if format_ == None:
        pass
    elif format_ == 'csv':
        data.to_csv(IdDevice + '_'+ start_  +'_' + end_ + '_ ' + frecuency  +'.csv')
    elif format_ == 'xlsx':
        data.to_excel(IdDevice + '_'+ start_  +'_' + end_ + '_ ' + frecuency  +'.xlsx')
    else:
        print('El formato no es valido. Formatos validos: csv y xlsx')
            
    return data