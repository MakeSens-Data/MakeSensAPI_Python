"""Modulo diseñado para dar acceso a los datos y facilitar el análisis de estos"""
from datetime import datetime
import calendar
import json
from urllib.parse import urlencode
import requests
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import seaborn as sns
import pkg_resources

def download_data(id_device: str, start_date: str, end_date: str,
                    sample_rate: str =  None, logs:bool = False,
                    data_type:str = 'RAW',  file_format: str = None,
                    fields: str = None):
    """
    Descarga y procesa datos de un dispositivo en un rango de fechas especificado.

    Esta función descarga datos de un dispositivo utilizando la API de Makesens,
    procesa los datos descargados y devuelve un DataFrame. Si se proporciona un 
    formato, también guarda los datos en un archivo con ese formato.

    Args:
        id_device (str): ID del dispositivo desde el cual se descargan los datos.
        start_date (str): Fecha y hora de inicio en formato 'YYYY-MM-DD HH:MM:SS'.
        end_date (str): Fecha y hora de fin en formato 'YYYY-MM-DD HH:MM:SS'.
        sample_rate (str): Tasa de muestreo para los datos ('1T' para minutos, '1H'
            para horas, '1D' para días).
        logs (bool, optional): Indica si se quiere descargar los logs. Por defecto 
            False (descarga data)
        data_type (str, optional): Indica el tipo de dato que se va a descargar:
             RAW o PROCESSED. Por defecto es RAW
        file_format (str, optional): Formato para guardar los datos descargados 
            ('csv' o 'xlsx'). Por defecto None.
        fields (str, optional): Lista de campos específicos a descargar.
            Por defecto None (todos los campos).

    Returns:
        pd.DataFrame: DataFrame con los datos descargados.

    Ejemplo:
        >>> data = download_data('device123', '2023-01-01 00:00:00', 
                                '2023-01-02 00:00:00', '1H', 'csv', 'pm10_1')
    """
    # Convertir las fechas string a datetime
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    end_datetime = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

    # Convertir datetime a timestamp Unix
    start_timestamp_ms = int(calendar.timegm(start_datetime.utctimetuple())) * 1000
    end_timestamp_ms = int(calendar.timegm(end_datetime.utctimetuple())) * 1000

    # Transformar los campos para poder hacer la solicitud
    if fields is not None:
        fields = fields.split(',')
        fields = str(','.join(__convert_measurements(fields, mode='upper')))
    # Descargar los datos
    data:list = []
    initial_timestamp_ms = start_timestamp_ms
    params = {'min_ts':initial_timestamp_ms,
              'max_ts':end_timestamp_ms,
              'data_type': data_type
              }
    if sample_rate:
        params['agg'] = sample_rate 
    if fields is not None:
        params['fields'] = fields
    while initial_timestamp_ms < end_timestamp_ms:
        encoded_params = urlencode(params)
        url_type = 'logs' if logs else 'data'
        url = f'https://api.makesens.co/device/{id_device}/{url_type}?{encoded_params}'
        try:
            response = requests.get(url, timeout=30).content
            response_json = json.loads(response)
        except Exception as e:
            raise ValueError('Error fetching or parsing data') from e
        response_end_timestamp_ms = int(response_json['date_range']['end'])
        if len(response_json) == 1 or initial_timestamp_ms == response_end_timestamp_ms:
            break
        initial_timestamp_ms = response_end_timestamp_ms
        params['min_ts'] = initial_timestamp_ms
        data.extend(response_json['data'])
    if not data:
        raise ValueError("There are no data for that date range.")
    else:
        dataframe_data = pd.DataFrame(data)
        dataframe_data['ts'] = pd.to_datetime(dataframe_data['ts'], unit='ms', utc=False)
        dataframe_data.index = pd.DatetimeIndex(dataframe_data.ts)
        dataframe_data.drop(columns=['ts'], inplace=True)
        # Modificar los nombres de las variables
        new_columns = __convert_measurements(list(dataframe_data.columns))
        dataframe_data.columns = new_columns
        dataframe_data.rename(columns={
            "pm10_1_ae" : "pm10_1_AE",
            "pm10_2_ae" : "pm10_2_AE",
            "pm25_1_ae" : "pm25_1_AE",
            "pm25_2_ae" : "pm25_2_AE",
            "pm1_1_ae" : "pm1_1_AE",
            "pm1_2_ae" : "pm1_2_AE",		
        }, inplace=True)

        if sample_rate:
            dataframe_data = dataframe_data.resample(sample_rate).mean()
        if file_format is not None:
            start_datetime_str = start_datetime.strftime("%Y-%m-%d_%H_%M_%S")
            end_datetime_str = end_datetime.strftime("%Y-%m-%d_%H_%M_%S")
            name = f"{id_device}_{start_datetime_str}_{end_datetime_str}_{sample_rate}"
            __save_data(dataframe_data, name, file_format)
        return dataframe_data


def __save_data(data, name: str, file_format:str) -> None:
    """
    Guarda los datos en un archivo en el formato especificado.

    Esta función toma un DataFrame de datos, un nombre de archivo y un formato ('csv' o 'xlsx').
    Luego, guarda los datos en el formato especificado utilizando las funciones to_csv o to_excel.

    Args:
        data (pd.DataFrame): Los datos que se van a guardar.
        name (str): Nombre del archivo (sin la extensión).
        file_format (str): Formato del archivo ('csv' o 'xlsx').

    Returns:
        None

    Ejemplo:
        >>> df = pd.DataFrame({'column1': [1, 2, 3], 'column2': [4, 5, 6]})
        >>> __save_data(df, 'my_data', 'csv')
    """
    if file_format == 'csv':
        data.to_csv(name + '.csv')
    elif file_format == 'xlsx':
        data.to_excel(name + '.xlsx')
    else:
        print('Formato no válido. Formatos válidos: csv y xlsx')

def __convert_measurements(measurements: list[str], mode="lower"):
    """
    Convierte y corrige nombres de mediciones según un modo especificado.

    Esta función toma una lista de nombres de mediciones, opcionalmente 
    corrige algunos nombres según un diccionario de correcciones específicas y
    luego los convierte a mayúsculas o minúsculas según el modo especificado.

    Args:
        measurements (list[str]): Lista de nombres de mediciones.
        mode (str): Modo de conversión ('lower' para minúsculas, 'upper' para mayúsculas).

    Returns:
        list[str]: Lista de nombres de mediciones convertidos.

    Ejemplo:
        >>> measurements = ['temperatura2', 'HUMEDAD_2', 'NO2']
        >>> converted_measurements = __convert_measurements(measurements, 'upper')
    """
    # Diccionario de correcciones específicas
    corrections = {
        "temperatura2": "temperatura_2",
        "temperatura_2": "temperatura2",
        "humedad2": "humedad_2",
        "humedad_2": "humedad2",
        "TEMPERATURA2": "TEMPERATURA_2",
        "TEMPERATURA_2": "temperatura2", 
        "HUMEDAD2": "HUMEDAD_2",
        "HUMEDAD_2": "humedad2"
    }

    new_measurements = []

    for measurement in measurements:
        # Aplicar correcciones específicas si es necesario
        corrected_measurement = corrections.get(measurement, measurement)

        # Convertir a mayúsculas o minúsculas según el modo
        new_measurement = (corrected_measurement.upper()
                            if mode == 'upper'
                            else corrected_measurement.lower())
        new_measurements.append(new_measurement)

    return new_measurements

##### Gradient #####
def gradient(data: pd.Series, variable: str):
    """
    Genera un gráfico de línea con gradiente de color basado en los valores de una serie de pandas.
    
    La función interpola los valores de la serie de datos para crear un gráfico más suave.
    Los colores del gráfico varían según los valores de la serie, utilizando un gradiente
    de color personalizado.

    Args:
        data (pd.Series): Serie de pandas que contiene los datos a visualizar. El índice
        de la serie debe ser un objeto de tipo datetime.
        variable (str): Nombre de la variable que se visualizará, usado para obtener los 
        rangos de colores normalizados y las unidades para la etiqueta del eje Y.

    Nota:
        Esta función asume que `load_normalized_color_ranges` está definida y devuelve una 
        tupla con los colores normalizados, la escala de colores y las unidades de medida 
        de la variable.

    Efectos secundarios:
        Muestra un gráfico de línea con un gradiente de color que representa la variable
        especificada a lo largo del tiempo.
    """
    new_data = data.copy()
    new_data.index = new_data.index.strftime("%Y-%m-%d %H:%M:%S")

    colors, scale, units = __load_normalized_color_ranges(variable)
    newcmp = LinearSegmentedColormap.from_list('testCmap', colors=colors, N=256)

    original_values = data.values
    original_index = np.linspace(1, len(original_values), len(original_values))
    interpolation_index = np.linspace(1, len(original_values), len(original_values) * 100)
    interpolated_values = np.interp(interpolation_index, original_index, original_values)
    interpolated_points = np.array([interpolation_index - 1, interpolated_values]).T.reshape(-1,1,2) # pylint: disable=too-many-function-args
    line_segments = np.concatenate([interpolated_points[:-1], interpolated_points[1:]], axis=1)

    norm = plt.Normalize(scale[0], scale[1])
    colored_line_segments = LineCollection(line_segments, cmap=newcmp, norm=norm)

    fig, ax = plt.subplots(figsize=(15, 5))
    new_data.plot(lw=0)
    colored_line_segments.set_array(interpolated_values)
    colored_line_segments.set_linewidth(1)
    line = ax.add_collection(colored_line_segments)
    fig.colorbar(line, ax=ax)
    ax.set_ylim(min(interpolated_values) - 10, max(interpolated_values) + 10)
    plt.ylabel(f'{variable} [{units}]', fontsize=14)
    plt.xlabel('Estampa temporal', fontsize=14)
    plt.gcf().autofmt_xdate()
    plt.show()


##### Heatmap #####
def heatmap(data:pd.Series, variable:str):
    """
    Genera un heatmap de los valores promedio de una variable agrupados por día y hora.
    
    La función agrupa los valores de la serie `data` por día y hora, calcula el promedio de estos 
    valores en cada grupo y luego los visualiza en un heatmap, donde el eje Y representa las horas
    del día, el eje X representa los días y los colores representan los valores promedio de la 
    variable especificada.
    
    Args:
        data (pd.Series): Serie de Pandas que contiene los valores a visualizar. 
                          El índice de la serie debe ser un DateTimeIndex.
        variable (str): Nombre de la variable representada por `data`. Utilizado para obtener
                        los rangos de colores normalizados y las unidades para la etiqueta
                        del eje Y.
                        
    Nota:
        Esta función asume que `__load_normalized_color_ranges` está definida y devuelve una tupla
        con los colores normalizados, la escala de colores y las unidades de medida de la variable.
        
    Efectos secundarios:
        Muestra un heatmap usando Matplotlib y Seaborn. No devuelve ningún valor.
    """
    colors, scale, units = __load_normalized_color_ranges(variable)
    newcmp = LinearSegmentedColormap.from_list('newCmap', colors=colors, N=256)
    norm = plt.Normalize(scale[0], scale[1])

    grouped = data.groupby([data.index.date, data.index.hour]).mean()
    heatmap_data = grouped.unstack(level=-1)
    heatmap_data = heatmap_data.iloc[:,::-1]

    plt.figure(figsize=(10,8))
    sns.heatmap(heatmap_data.T, cmap=newcmp, norm=norm)
    plt.ylabel('Horas', fontsize=14)
    plt.xlabel('Estampa temporal', fontsize=14)
    plt.show()

def __load_normalized_color_ranges(variable_name: str) -> list:
    """
    Carga y normaliza los rangos de colores para una variable específica desde 
    un archivo JSON.

    La función abre y lee un archivo JSON que contiene rangos de colores asociados
    a diferentes variables. Luego, normaliza los rangos de la variable especificada 
    entre 0 y 1, manteniendo los colores asociados.

    Args:
        variable_name (str): El nombre de la variable para la cual se cargarán y
        normalizarán los rangos de colores.

    Returns:
        list: Una lista de tuplas, donde cada tupla contiene un rango normalizado 
        (entre 0 y 1) y el color asociado.
    """
    # Abrir y cargar los datos desde el archivo JSON
    file_path = pkg_resources.resource_filename('MakeSens', 'colors_by_variable.json')
    with open(file_path, 'r', encoding="utf-8") as file:
        data = json.load(file)


    # Extraer los rangos de colores para la variable especificada
    color_ranges = data[variable_name]['ranges']
    units= data[variable_name]['units']

    # Extraer solo los rangos (primer elemento de cada tupla)
    ranges = [range_value for range_value, _ in color_ranges]

    # Normalizar los rangos entre 0 y 1
    min_range = min(ranges)
    max_range = max(ranges)
    scale = (min_range, max_range)
    normalized_ranges = [
        (range_value - min_range) / (max_range - min_range)
        for range_value in ranges
    ]


    # Asociar los rangos normalizados con los colores correspondientes
    normalized_color_ranges = [
        (normalized_range, color)
        for normalized_range, (_,color) in zip(normalized_ranges, color_ranges)
    ]

    return normalized_color_ranges, scale, units
# End-of-file