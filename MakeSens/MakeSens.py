import pandas as pd
import requests
import json
from datetime import datetime
import calendar
from typing import List, Tuple
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from urllib.parse import urlencode



def __save_data(data, name: str, format: str) -> None:
    """
    Guarda los datos en un archivo con el formato especificado.

    Esta función toma un DataFrame de pandas y lo guarda en un archivo con el formato
    especificado (csv o xlsx) utilizando el nombre proporcionado.

    Args:
        data (pd.DataFrame): Los datos que se guardarán en el archivo.
        name (str): El nombre base del archivo (la extensión se agrega automáticamente).
        format (str): El formato del archivo de salida ('csv' o 'xlsx').

    Returns:
        None

    Raises:
        ValueError: Si el formato no es 'csv' ni 'xlsx'.

    Ejemplo:
        >>> data = pd.DataFrame({'columna1': [1, 2, 3], 'columna2': ['A', 'B', 'C']})
        >>> __save_data(data, 'mi_archivo', 'csv')  # Guarda los datos en mi_archivo.csv
    """

    if format == 'csv':
        data.to_csv(name + '.csv', index=False)  # Guarda los datos en formato CSV sin índice.
    elif format == 'xlsx':
        data.to_excel(name + '.xlsx', index=False)  # Guarda los datos en formato Excel sin índice.
    else:
        raise ValueError("Formato no válido. Formatos válidos: 'csv' y 'xlsx'")



def __convert_measurements(measurements: list[str], mode: str = "lower") -> list[str]:
    """
    Convierte y corrige medidas en una lista de acuerdo al modo especificado.

    Esta función toma una lista de medidas y opcionalmente un modo ('upper' o 'lower')
    y devuelve una nueva lista de medidas convertidas en mayúsculas o minúsculas.

    Args:
        measurements (list[str]): La lista de medidas a convertir.
        mode (str, opcional): El modo de conversión ('upper' para mayúsculas o 'lower' para minúsculas).
                              Por defecto, se convierte a minúsculas.

    Returns:
        list[str]: Una nueva lista de medidas convertidas según el modo especificado.

    Ejemplo:
        >>> measures = ['temperatura2', 'HUMEDAD_2', 'presion3']
        >>> __convert_measurements(measures, 'upper')
        ['TEMPERATURA2', 'HUMEDAD_2', 'PRESION3']
    """
    
    # Diccionario de correcciones específicas
    corrections = {
        "temperatura2": "temperatura_2",
        "temperatura_2": "temperatura2",
        "humedad2": "humedad_2",
        "humedad_2": "humedad2",
        "TEMPERATURA2": "TEMPERATURA_2",
        "TEMPERATURA_2": "TEMPERATURA2",  # Nota: revisa si realmente es "TEMPERATUA2"
        "HUMEDAD2": "HUMEDAD_2",
        "HUMEDAD_2": "HUMEDAD2"
    }

    new_measurements = []

    for measurement in measurements:
        # Aplicar correcciones específicas si es necesario
        measurement = corrections.get(measurement, measurement)

        # Convertir a mayúsculas o minúsculas según el modo
        new_measurement = measurement.upper() if mode == 'upper' else measurement.lower()
        new_measurements.append(new_measurement)

    return new_measurements



def download_data(id_device: str, start_date: str, end_date: str, sample_rate: str, format: str = None, fields: str = None) -> pd.DataFrame:
    """
    Descarga datos de un dispositivo desde una API, los procesa y opcionalmente los guarda en un archivo.

    Esta función descarga datos de un dispositivo a través de una API, los procesa para convertir las fechas
    y las variables, y devuelve un DataFrame de pandas con los datos descargados. Los datos también pueden ser
    guardados en un archivo con un formato específico si se proporciona.

    Args:
        id_device (str): ID del dispositivo desde el cual se descargan los datos.
        start_date (str): Fecha y hora de inicio en formato 'YYYY-MM-DD HH:MM:SS'.
        end_date (str): Fecha y hora de fin en formato 'YYYY-MM-DD HH:MM:SS'.
        sample_rate (str): Tasa de muestreo para la agregación de datos.
        format (str, opcional): Formato de archivo para guardar los datos (opciones: 'csv' o 'xlsx').
        fields (str, opcional): Lista de variables separadas por comas para descargar y procesar.

    Returns:
        pd.DataFrame: DataFrame que contiene los datos descargados y procesados.

    Ejemplo:
        >>> data = download_data('mE1_00003', '2023-01-01 00:00:00', '2023-01-02 00:00:00', '1h', 'csv', 'temperature,humidity')
    """
    # Convertir las fechas string a datetime
    start_date_ = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    end_date_ = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

    # Convertir datetime a timestamp Unix
    start = int(calendar.timegm(start_date_.utctimetuple()))
    end = int(calendar.timegm(end_date_.utctimetuple()))

    dat = []  # Almacenar los datos
    tmin = start

    if fields is not None:
        fields = fields.split(',')
        fields = ','.join(__convert_measurements(fields, mode='upper'))

    while tmin < end:
        params = {'min_ts': tmin,
                  'max_ts': end,
                  'agg': sample_rate}

        if fields is not None:
            params['fields'] = fields

        encoded_params = urlencode(params)
        url = f'https://api.makesens.co/device/{id_device}/data?{encoded_params}'
        try:
            rta = requests.get(url).content
            d = json.loads(rta)
        except Exception as e:
            print(f"Error fetching or parsing data: {e}")
            break

        # Salir del bucle si no hay datos o si el timestamp no ha cambiado
        if len(d) == 1 or tmin == int(d['date_range']['end']):
            break

        tmin = int(d['date_range']['end'])
        dat.extend(d['data'])

    if dat:
        data = pd.DataFrame(dat)
        data['ts'] = pd.to_datetime(data['ts'], unit='ms', utc=False)

        # Poner las variables como las conoce el
        new_columns = __convert_measurements(list(data.columns))
        data.columns = new_columns

        start_ = start_date.replace(':', '_')
        end_ = end_date.replace(':', '_')
        name = id_device + '_' + start_ + '_' + end_ + '_ ' + sample_rate

        if format is not None:
            __save_data(data, name, format)

        return data



# def download_data(id_device:str,start_date:str,end_date:str, sample_rate:str,format:str = None, fields:str = None):
    
#     start:int = int((datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S") - datetime(1970, 1, 1)).total_seconds())
#     end:int = int((datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S") -  datetime(1970, 1, 1)).total_seconds())
    
#     dat:list = []
#     tmin:int = start
        
#     while tmin < end:
#         if fields == None:
#             url = f'https://api.makesens.co/ambiental/metricas/{id_device}/data?agg=1{sample_rate}&agg_type=mean&items=1000&max_ts={str(end * 1000)}&min_ts={str(tmin * 1000)}'
#         else:
#             url =  f'https://api.makesens.co/ambiental/metricas/{id_device}/data?agg=1{sample_rate}&agg_type=mean&fields={fields}&items=1000&max_ts={str(end * 1000)}&min_ts={str(tmin * 1000)}'
#         rta = requests.get(url).content
#         d = json.loads(rta)
#         try:
#             if tmin == (d[-1]['ts']//1000) + 1:
#                 break
#             dat = dat + d
#             tmin = (d[-1]['ts']//1000) + 1
#         except IndexError:
#             break
       
#     data = pd.DataFrame([i['val'] for i in dat], index=[datetime.utcfromtimestamp(i['ts']/1000).strftime('%Y-%m-%d %H:%M:%S') for i in dat])
    
#     start_ = start_date.replace(':','_') 
#     end_ = end_date.replace(':','_')
#     name = id_device + '_'+ start_  +'_' + end_ + '_ ' + sample_rate
    
#     if format != None:
#         __save_data(data,name,format)    
    
        
#     return data



# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------


def __gradient_plot(data, scale, y_label, sample_rate):
    """
    Crea un gráfico de gradiente a partir de los datos proporcionados.

    Esta función toma datos de un DataFrame, un rango de escala, una etiqueta para el eje y y una tasa de muestreo.
    Luego, crea un gráfico de gradiente interpolando los datos para ajustarlos a la escala y aplicando una paleta de colores.

    Args:
        data (pd.DataFrame): Los datos para generar el gráfico.
        scale (tuple): Rango de escala para el color del gradiente.
        y_label (str): Etiqueta para el eje y del gráfico.
        sample_rate (str): Tasa de muestreo ('m' para minutos, 'w' para semanas).

    Returns:
        None

    Ejemplo:
        >>> data = pd.DataFrame({'date': ['2023-08-01', '2023-08-02', '2023-08-03'],
        ...                      'PM': [10, 20, 15]})
        >>> __gradient_plot(data, (0, 30), 'Partículas', 'm')
    """
    # Convertir la tasa de muestreo a frecuencia pandas
    if sample_rate == 'm':
        sample_rate = 'T'
    elif sample_rate == 'w':
        sample_rate = '7d'

    # Crear un rango de fechas y llenar con NaN si no hay datos
    data.index = pd.DatetimeIndex(data.index)
    date_range = pd.date_range(data.index[0], data.index[-1], freq=sample_rate)
    values = []
    for date in date_range:
        if date in data.index:
            values.append(data.loc[date, 'PM'])
        else:
            values.append(np.nan)

    # Crear un DataFrame con los datos interpolados
    interpolated_data = pd.DataFrame(index=date_range)
    interpolated_data['PM'] = values
    interpolated_data.index = interpolated_data.index.strftime("%Y-%m-%d %H:%M:%S")

    # Definir la paleta de colores
    colorlist = ["green", "yellow", 'orange', "red", 'purple', 'brown']
    newcmp = LinearSegmentedColormap.from_list('testCmap', colors=colorlist, N=256)

    y_values = np.array(list(interpolated_data['PM']))
    x_values = np.linspace(1, len(y_values), len(y_values))

    x_range = np.linspace(1, len(y_values), 10000)
    y_range = np.interp(x_range, x_values, y_values)

    points = np.array([x_range - 1, y_range]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, ax = plt.subplots(figsize=(15, 5))

    norm = plt.Normalize(scale[0], scale[1])
    lc = LineCollection(segments, cmap=newcmp, norm=norm)

    lc.set_array(y_range)
    lc.set_linewidth(1)
    line = ax.add_collection(lc)
    interpolated_data['PM'].plot(lw=0)
    plt.colorbar(line, ax=ax)
    ax.set_ylim(min(y_range) - 10, max(y_range) + 10)
    plt.ylabel(y_label + ' $\mu g / m^3$', fontsize=14)
    plt.xlabel('Estampa temporal', fontsize=14)
    plt.gcf().autofmt_xdate()
    plt.show()


def gradient_pm10(id_device: str, start_date: str, end_date: str, sample_rate: str):
    """
    Descarga, procesa y visualiza los datos PM10 de un dispositivo en un gráfico de gradiente.

    Esta función descarga los datos PM10 de un dispositivo en el período especificado, los procesa y crea un gráfico
    de gradiente utilizando la función '__gradient_plot'. La escala y la tasa de muestreo se configuran según la necesidad.

    Args:
        id_device (str): ID del dispositivo desde el cual se descargan los datos.
        start_date (str): Fecha y hora de inicio en formato 'YYYY-MM-DD HH:MM:SS'.
        end_date (str): Fecha y hora de fin en formato 'YYYY-MM-DD HH:MM:SS'.
        sample_rate (str): Tasa de muestreo ('m' para minutos, 'w' para semanas).

    Returns:
        None

    Ejemplo:
        >>> gradient_pm10('mE1_00003', '2023-01-01 00:00:00', '2023-01-02 00:00:00', '1h')
    """
    # Descargar los datos PM10
    data = download_data(id_device, start_date, end_date, sample_rate, fields='pm10_1')
    
    # Eliminar duplicados de tiempo y reorganizar el DataFrame
    data['ts'] = data.index
    data = data.drop_duplicates(subset=['ts'])
    
    # Crear el gráfico de gradiente
    __gradient_plot(data['pm10_1'], (54, 255), 'PM10 ', sample_rate)


def gradient_pm2_5(id_device: str, start_date: str, end_date: str, sample_rate: str):
    """
    Descarga, procesa y visualiza los datos PM2.5 de un dispositivo en un gráfico de gradiente.

    Esta función descarga los datos PM2.5 de un dispositivo en el período especificado, los procesa y crea un gráfico
    de gradiente utilizando la función '__gradient_plot'. La escala y la tasa de muestreo se configuran según la necesidad.

    Args:
        id_device (str): ID del dispositivo desde el cual se descargan los datos.
        start_date (str): Fecha y hora de inicio en formato 'YYYY-MM-DD HH:MM:SS'.
        end_date (str): Fecha y hora de fin en formato 'YYYY-MM-DD HH:MM:SS'.
        sample_rate (str): Tasa de muestreo ('m' para minutos, 'w' para semanas).

    Returns:
        None

    Ejemplo:
        >>> gradient_pm2_5('mE1_00003', '2023-01-01 00:00:00', '2023-01-02 00:00:00', '1h')
    """
    # Descargar los datos PM2.5
    data = download_data(id_device, start_date, end_date, sample_rate, fields='pm25_1')
    
    # Eliminar duplicados de tiempo y reorganizar el DataFrame
    data['ts'] = data.index
    data = data.drop_duplicates(subset=['ts'])
    
    # Crear el gráfico de gradiente
    __gradient_plot(data['pm25_1'], (12, 251), 'PM2.5 ', sample_rate)


# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------


def _heatmap_plot(data, scale, title):
    """
    Crea un gráfico de mapa de calor a partir de los datos proporcionados.

    Esta función toma datos de un DataFrame, un rango de escala y un título, y crea un gráfico de mapa de calor
    utilizando la librería Seaborn. El color del mapa de calor se ajusta a la escala proporcionada.

    Args:
        data (pd.Series): Los datos para generar el gráfico.
        scale (tuple): Rango de escala para el color del mapa de calor.
        title (str): Título del gráfico.

    Returns:
        None

    Ejemplo:
        >>> data = pd.Series([10, 20, 30, 15, 25, 35], index=pd.date_range('2023-08-01', periods=6, freq='H'))
        >>> _heatmap_plot(data, (10, 35), 'Valores de PM2.5 por hora')
    """
    # Definir la paleta de colores y la normalización
    colorlist = ["green", "yellow", 'orange', "red", 'purple', 'brown']
    newcmp = LinearSegmentedColormap.from_list('testCmap', colors=colorlist, N=256)
    norm = plt.Normalize(scale[0], scale[1])

    # Crear un DataFrame para el mapa de calor
    date = pd.date_range(data.index.date[0], data.index.date[-1]).date
    hours = range(0, 24)
    mapa = pd.DataFrame(columns=date, index=hours, dtype="float")

    # Llenar el DataFrame del mapa de calor con los datos
    for i in range(0, len(date)):
        dat = data[data.index.date == date[i]]
        for j in range(0, len(dat)):
            fila = dat.index.hour[j]
            mapa[date[i]][fila] = dat[j]

    # Crear y mostrar el gráfico de mapa de calor
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(mapa, cmap=newcmp, norm=norm)
    plt.ylabel('Horas', fontsize=16)
    plt.xlabel('Estampa temporal', fontsize=16)
    plt.title(title + ' $\mu g / m^3$', fontsize=16)
    plt.show()

    

def heatmap_pm10(id_device: str, start_date: str, end_date: str):
    """
    Descarga, procesa y visualiza un mapa de calor para los datos PM10 de un dispositivo.

    Esta función descarga los datos PM10 de un dispositivo en el período especificado, los procesa y crea un gráfico
    de mapa de calor utilizando la función '_heatmap_plot'. La escala y el título se configuran según la necesidad.

    Args:
        id_device (str): ID del dispositivo desde el cual se descargan los datos.
        start_date (str): Fecha y hora de inicio en formato 'YYYY-MM-DD HH:MM:SS'.
        end_date (str): Fecha y hora de fin en formato 'YYYY-MM-DD HH:MM:SS'.

    Returns:
        None

    Ejemplo:
        >>> heatmap_pm10('mE1_00003', '2023-01-01 00:00:00', '2023-01-02 00:00:00')
    """
    # Descargar los datos PM10
    data = download_data(id_device, start_date, end_date, 'h', fields='pm10_1')
    
    # Establecer el índice y seleccionar los datos PM10
    data.index = pd.DatetimeIndex(data.index)
    data = data['pm10_1']
    
    # Crear el gráfico de mapa de calor
    _heatmap_plot(data, (54, 255), 'PM10')

    
def heatmap_pm2_5(id_device: str, start_date: str, end_date: str):
    """
    Descarga, procesa y visualiza un mapa de calor para los datos PM2.5 de un dispositivo.

    Esta función descarga los datos PM2.5 de un dispositivo en el período especificado, los procesa y crea un gráfico
    de mapa de calor utilizando la función '_heatmap_plot'. La escala y el título se configuran según la necesidad.

    Args:
        id_device (str): ID del dispositivo desde el cual se descargan los datos.
        start_date (str): Fecha y hora de inicio en formato 'YYYY-MM-DD HH:MM:SS'.
        end_date (str): Fecha y hora de fin en formato 'YYYY-MM-DD HH:MM:SS'.

    Returns:
        None

    Ejemplo:
        >>> heatmap_pm2_5('mE1_00003', '2023-01-01 00:00:00', '2023-01-02 00:00:00')
    """
    # Descargar los datos PM2.5
    data = download_data(id_device, start_date, end_date, 'h', fields='pm25_1')
    
    # Establecer el índice y seleccionar los datos PM2.5
    data.index = pd.DatetimeIndex(data.index)
    data = data['pm25_1']
    
    # Crear el gráfico de mapa de calor
    _heatmap_plot(data, (12, 251), 'PM2.5')


# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------

def weekly_profile(id_device: str, start_date: str, end_date: str, field: str):
    """
    Visualiza el perfil semanal de una variable en un gráfico con barras de error.

    Esta función descarga los datos de una variable específica de un dispositivo en el período especificado,
    calcula el perfil semanal promedio y la desviación estándar para cada hora del día, y crea un gráfico
    para visualizar el perfil semanal con barras de error. El título del gráfico se personaliza según la variable.

    Args:
        id_device (str): ID del dispositivo desde el cual se descargan los datos.
        start_date (str): Fecha y hora de inicio en formato 'YYYY-MM-DD HH:MM:SS'.
        end_date (str): Fecha y hora de fin en formato 'YYYY-MM-DD HH:MM:SS'.
        field (str): Nombre de la variable ('PM10', 'PM2.5' o 'CO2').

    Returns:
        None

    Ejemplo:
        >>> weekly_profile('device123', '2023-01-01 00:00:00', '2023-01-07 23:59:59', 'PM10')
    """
    fields = {
        'PM10': {'variable': 'pm10_1', 'unidades': '[$\mu g/m^3$]'},
        'PM2.5': {'variable': 'pm25_1', 'unidades': '[$\mu g/m^3$]'},
        'CO2': {'variable': 'ppm', 'unidades': '[ppm]'}
    }

    var = fields[field]['variable']
    unidad = fields[field]['unidades']

    data = download_data(id_device, start_date, end_date, 'h', fields=var)
    data.index = pd.DatetimeIndex(data.index)
    days = range(0, 7)
    hours = range(0, 24)

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
    plt.figure(figsize=(18, 4))
    plt.plot(x, np.array(variable_mean))
    plt.fill_between(x, np.array(variable_mean) - np.array(variable_std), np.array(variable_mean) + np.array(variable_std), color='r', alpha=0.2)
    plt.xticks(np.linspace(0, 162, 28), ['0', '6', '12', '18'] * 7)

    plt.hlines(b + 5, 0, 167, color='k')
    for i in np.linspace(0, 168, 8)[1:-1]:
        plt.vlines(i, a, b + 15, color='k', ls='--', lw=1)

    name_days = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    position_x = [9, 30, 55, 80, 103, 128, 150]
    for i in range(0, len(name_days)):
        plt.text(position_x[i], b + 8, name_days[i], fontsize=13)

    plt.xlim(0, 167)
    plt.ylim(a, b + 15)

    plt.xlabel('Horas', fontsize=14)
    plt.ylabel(f'{field} {unidad}', fontsize=14)
    plt.title(f'Perfil Semanal de {field}', fontsize=16)
    plt.show()
  
