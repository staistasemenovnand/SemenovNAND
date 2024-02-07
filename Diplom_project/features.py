#------------- ПОДКЛЮЧЕНИЕ БИБЛИОТЕК--------------------------------------------------------------
import pika
import json
import traceback
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import datetime
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
import pickle
import geonamescache
import time
import re
import math
import statistics
from haversine import haversine, Unit


#------------- ЗАГРУЗКА ЗНАЧЕНИЙ ПО УМОЛЧАНИЮ И СПРАВОЧНИКОВ---------------------------------------
with open("components/default_values.pickle", "rb") as dv:
    defaults = pickle.load(dv)
    
with open("components/model_features.pickle", "rb") as fm:
    model_features = pickle.load(fm)

with open("components/rent_model_features.pickle", "rb") as fr:
    rent_model_features = pickle.load(fr)
    
with open("components/main_scaler.pickle", "rb") as ms:
    main_scaler = pickle.load(ms)
    
with open("components/rent_scaler.pickle", "rb") as rs:
    rent_scaler = pickle.load(rs)
    
common_medians, schools_medians_city, schools_medians_all, \
hometype_medians_sqft, states_capitals, other_states, \
rent_features_for_del, features_for_del = defaults

gc = geonamescache.GeonamesCache(min_city_population=500)

with open("components/us_cities.pickle", "rb") as f:
    us_cities = pickle.load(f)

#------------- ПАКЕТ ФУНКЦИЙ ДЛЯ ПРЕДОБРАБОТКИ ДАННЫХ И ПОЛУЧЕНИЯ ПРЕДСКАЗАНИЯ МОДЕЛИ--------------



# 0. для очистки признака 'status'------------------------------------------------------------

def normalise_status(status):
    
# cформируем списки статусов-синонимов для замены по точному совпадению
    ordinary = ['for sale','active']
    coming_soon = ['coming soon']
    pending = ['p','ps', 'pf', 'pi']
    foreclosure = ['foreclos']
    active = ['a active','listing extended','back on market','re activated','reactivated','price change']
    new_construction = ['new']
    contingency = ['c', 'c continue show','conting accpt backups','ct']
    under_contract = ['uc continue to show','due diligence period','accepted offer','accepting backups','backup']
    for_rent = ['lease/purchase']

# сначала сделаем замены по like-условиям:
    if status != status: return 'Zero status'
    if status is None: return 'Zero status'
    if 'coming soon' in status: return 'coming soon'
    if 'pending' in status: return 'pending'
    if 'foreclos' in status: return 'foreclosure'
    if 'contract' in status: return 'under contract'
    if 'contingen' in status: return 'contingency'
    if 'auction' in status: return 'auction'
    if 'rent' in status: return 'for rent'
    if 'sold' in status: return 'sold'
    if 'closed' in status: return 'sold'
    if 'active' in status: return 'for sale'
    
# теперь сделаем замены по точным совпадениям из словарей:
    if status in pending: return 'pending'
    if status in active: return 'active'
    if status in contingency: return 'contingency'
    if status in new_construction: return 'new construction'
    if status in under_contract: return 'under contract'
    if status in for_rent: return 'for rent'
    if status in ordinary: return 'for sale'
    return status



# 1. Для очиски признака 'private pool'----------------------------------------------------

def normalise_pool(df):
    if df['PrivatePool'] == 'yes':
        return 'Yes'
    else:
        return df['private pool']
    
    
    
# 2. Для очистки признака 'city'-----------------------------------------------------------
    
def get_city(df):
    city = df['city']
    state = df['state']
# если на входе не указан город или штат - возвращаем ответ о некорректном входе
    if len(city)==0 or city.isspace() or len(state)==0 or state.isspace():
        return pd.Series(['Empty input','Empty input',0, 0, 0])
# если на входе и город, и штат, идем искать
    else:
        search = gc.search_cities(city, case_sensitive=False, contains_search=False)
# если не нашли такой город - возвращаем ответ 'Not found'
        if not search:
            return pd.Series(['Not found','Not found',0, 0, 0])
# если нашли, то в списке ищем запись с совпадающим штатом
        else:
            for i in search:
# если нашли таковую - возвращаем результат
                if i['admin1code'] == state:
                    return pd.Series([i['geonameid'], i['name'], i['population'], i['latitude'], i['longitude']])
# иначе возвращаем ответ о ненайденном соответствии по штату
            return pd.Series(['Not matched state','Not matched state', 0, 0, 0])
    return pd.Series(['Other error','Other error', 0, 0, 0])

def get_city_full(df, attempt = 1):
    
# получаем параметры для поиска
    state = df['state']
    geonameid = df['geonameid']
    attempt = attempt
    population = df['population']
    latitude = df['latitude']
    longitude = df['longitude']
    
# при первой попытке ищем исходное название города, при последующих - синоним
    if attempt == 1:
        city = df['city']
    if attempt > 1:
        city = df['city_new']
        
# зафиксируем перечень отрицательных результатов
    failed_id = ['Not found','Not matched state','Zero pop not found',
                 'Some zero pop found','Some pop found', 'Pop not found']
    pattern = '|'.join(failed_id)
    
# на первой и второй попытках, если ранее город уже идентифицирован - повторную идентификацию не проводим
    if attempt in (1,2) and (geonameid in failed_id) == False:
        result = pd.Series([geonameid, city, population, latitude, longitude])
        return result

# на третьей попытке проводим повторную идентификацию только если на предыдущих попытках нашли несколько городов
    if attempt == 3 and (geonameid in failed_id) and geonameid != 'Some pop found':
        result = pd.Series(['OTHER CITY-1','OTHER CITY-1', 0, 0, 0])
        return result    
    result = pd.Series(['Other error', 'Other error', 0, 0, 0])

# если на входе не указан город или штат - возвращаем ответ о некорректном входе
    if len(city)==0 or city.isspace() or len(state)==0 or state.isspace():
        result = pd.Series(['Empty input','Empty input', 0, 0, 0])
        return result

# если на входе и город, и штат, идем искать
# сначала пробуем найти по точному совпадению город с ненулевым населением и соспадающим штатом
    found_full = us_cities[(us_cities['name']==city) & (us_cities['admin1code']==state)
                              & (us_cities['feature class']=='P') & (us_cities['population'] > 0)]

# если нашли всего одну запись - возвращаем данные из нее
    if found_full.shape[0] ==1:
        result = pd.Series([found_full['geonameid'].values[0], found_full['name'].values[0],
                            found_full['population'].values[0], found_full['latitude'].values[0],
                            found_full['longitude'].values[0]])

        return result

# если не нашли ранее, то теперь ищем точное совпадение города и штата и населенный пункт с нулевой численностью
    found_full = us_cities[(us_cities['name']==city) & (us_cities['admin1code']==state)
                              & (us_cities['feature class']=='P') & (us_cities['population'] == 0)]
    
# если нашли всего одну запись - возвращаем данные из нее
    if found_full.shape[0] ==1:
        #(city,state,'OKF2')
        result = pd.Series([found_full['geonameid'].values[0], found_full['name'].values[0],
                            found_full['population'].values[0], found_full['latitude'].values[0],
                            found_full['longitude'].values[0]])
        return result   

# если не нашли ранее, то пытаемся найти уже совпадение по альтернативным именам города (они есть в базе городов США)
    else:
        found_city = us_cities[(us_cities['name'] == city) | (us_cities['alternatenames'].str.contains(city, na=False))]
        
# если не нашли такой город - возвращаем ответ 'Not found'
        if found_city.shape[0] == 0:
            result = pd.Series(['Not found','Not found', 0, 0, 0])
            return result

# если нашли, то в списке ищем запись с совпадающим штатом
        else:
            found_state = found_city[(found_city['admin1code'] == state)]
            
# если не нашли совпадение со штатом - возвращаем ответ 'Not matched state'
            if found_state.shape[0] == 0:
                result = pd.Series(['Not matched state','Not matched state', 0, 0 ,0])
                return result
            else:
                
# если нашли всего одну запись - возвращаем значения из нее
                if found_state.shape[0] == 1:
                    result = pd.Series([found_state['geonameid'].values[0], found_state['name'].values[0],
                                        found_state['population'].values[0], found_state['latitude'].values[0],
                                        found_state['longitude'].values[0]])
                    return result
            
# если нашли более одной записи - смотрим на численность населения
                else:
                    found_pop = found_state[(found_state['population'] > 0)]
            
# если нет населенных пунктов с населением больше 0, возвращаем ответ 'Pop not found'
                    if found_pop.shape[0] == 0:
                        found_zero_pop = found_city[(found_city['population'] == 0)]
                        if found_zero_pop.shape[0] == 0:
                            result = pd.Series(['Pop not found', 'Pop not found', 0, 0, 0])
                            return result
                        
# если найден всего один населенный пункт с численностью населению больше 0 - возвращаем данные из него
                        else:
                            if found_zero_pop.shape[0] == 1:
                                result = pd.Series([found_zero_pop['geonameid'].values[0], found_zero_pop['name'].values[0],
                                                   found_zero_pop['population'].values[0], found_zero_pop['latitude'].values[0],
                                                   found_zero_pop['longitude'].values[0]])
                                return result
                        
# если не нашли ранее - пытаемся найти населенный пункт с нулевой численностью населения
                            else:
                                found_zero_pop_fp = found_zero_pop[(found_zero_pop['feature class'] == 'P')]
            
# если не нашли - возвращаем ответ 'Zero popmnot found'
                                if found_zero_pop_fp.shape[0] == 0:
                                    result = pd.Series(['Zero pop not found', 'Zero pop not found', 0, 0, 0])
                                    return result

# если нашли всего одну запись с нулевой численностью населения - возвращаем данные из нее
                                else:
                                    if found_zero_pop_fp.shape[0] == 1:
                                        result = pd.Series([found_zero_pop_fp['geonameid'].values[0], found_zero_pop_fp['name'].values[0],
                                                            found_zero_pop_fp['population'].values[0], found_zero_pop_fp['latitude'].values[0],
                                                            found_zero_pop_fp['longitude'].values[0]])
                                        return result
                    
# если нашли более одной записи с нулевой численностью - возвращаем ответ 'Some zero pop found'
                                    else:
                                        result = pd.Series(['Some zero pop found', 'Some zero pop found', 0, 0, 0])
                                        return result 
            
# если ранее мы нашли всего одну запись с населением больше 0, то возвращаем данные из нее
                    else:
                        if found_pop.shape[0] == 1:
                            result = pd.Series([found_pop['geonameid'].values[0], found_pop['name'].values[0],
                                                found_pop['population'].values[0], found_pop['latitude'].values[0],
                                                found_pop['longitude'].values[0]])
                            return result
                        
# если нашли более одной записи с населением больше 0, по проверяем, есть ли запись с признаком, что это именно населенный
# пункт: в используемой базе не только населенные пункты, но и туристичсекие места, 'местечки', объекты местности
                        else:
                            found_pop_fp = found_pop[(found_pop['feature class'] == 'P')] 
            
# если не нашли - возвращаем ответ 'Pop not found'
                            if found_pop_fp.shape[0] == 0:
                                result = pd.Series(['Pop not found', 'Pop not found', 0, 0, 0])
                                return result
                            else:
                        
# а если нашли всего 1 запись - возвращаем данные из нее
                                if found_pop_fp.shape[0] == 1:
                                    result = pd.Series([found_pop_fp['geonameid'].values[0], found_pop_fp['name'].values[0],
                                                       found_pop_fp['population'].values[0], found_pop_fp['latitude'].values[0],
                                                       found_pop_fp['longitude'].values[0]])
                                    return result
            
# если нашли более, чем одну запись - то на третьей попытке выбираем город с наибольшей численностью из найденных
                                else:

                                    if attempt == 3:
                                        found_pop_fp = found_pop_fp[found_pop_fp['population'] == found_pop_fp['population'].max()]
                                        result = pd.Series([found_pop_fp['geonameid'].values[0], found_pop_fp['name'].values[0],
                                                       found_pop_fp['population'].values[0], found_pop_fp['latitude'].values[0],
                                                       found_pop_fp['longitude'].values[0]])
                                    else:
                                        result = pd.Series(['Some pop found', 'Some pop found', 0, 0, 0])
                                        return result 
    return result

def get_cities_allias(df):
    failed_id = ['Not found','Not matched state','Zero pop not found','Some zero pop found','Some pop found','Pop not found','Not found pop','Not matched state']
    pattern = '|'.join(failed_id)
    df['geonameid'] = df['geonameid'].astype(str)
    df_for_replace = df[df['geonameid'].str.contains(pattern, na=False)]
    df_not_replace = df[~df['geonameid'].str.contains(pattern, na=False)]
    df_not_replace['city_new'] = df_not_replace['city']
    df_for_replace['city_new'] = df_for_replace['city'].str.replace(' TOWNSHIP', '')
    df_for_replace['city_new'] = df_for_replace['city_new'].str.replace(' TWP', '')
    df_for_replace['city_new'] = df_for_replace['city_new'].str.replace(' CITY', '')
    df_for_replace['city_new'] = df_for_replace['city_new'].str.replace(' TWP.', '')
    df_for_replace['city_new'] = df_for_replace['city_new'].str.replace(' PT', '')
    df_for_replace['city_new'] = df_for_replace['city_new'].str.replace(' VLG', '')
    df_for_replace['city_new'] = df_for_replace['city_new'].str.replace(' VL', '')
    df_for_replace['city_new'] = df_for_replace['city_new'].str.replace(' VILLAGE', '')
    df_for_replace['city_new'] = df_for_replace['city_new'].str.replace(' SPGS', '')
    df_for_replace['city_new'] = df_for_replace['city_new'].str.replace(' SPRINGS', '')
    df_for_replace['city_new'] = df_for_replace['city_new'].str.replace(' MANOR', '')
    df_for_replace['city_new'] = df_for_replace['city_new'].str.replace('ST ', 'SAINT ')
    df_for_replace['city_new'] = df_for_replace['city_new'].str.replace('FT ', 'FORT ')
    df_for_replace['city_new'] = df_for_replace['city_new'].str.replace('FT. ', 'FORT ')
    df_for_replace['city_new'] = df_for_replace['city_new'].str.replace(' HTS', '')
    df_for_replace['city_new'] = df_for_replace['city_new'].str.replace('PT ', 'PORT ')
    df_for_replace['city_new'] = df_for_replace['city_new'].str.replace(' LK', ' LAKE')
    df_for_replace['city_new'] = df_for_replace['city_new'].str.replace(' SHRS', '')
    df_for_replace['city_new'] = df_for_replace['city_new'].str.replace(' PKWY', '')
    df_for_replace['city_new'] = df_for_replace['city_new'].str.replace(' VALLEY', '')

    cities_allias = {'ATLAANTA' : 'ATLANTA',
                 'BAYSIDE HILLS' : 'BAYSIDE',
                 'BELLLINGHAM' : 'BELLINGHAM',
                 'BOSTON (DORCHESTER)' : 'BOSTON',
                 'BRONX' : 'NEW YORK CITY',
                 'CHERRYHILLSVILLAGE' : 'CHERRY HILLS VILLAGE',
                 'COMMERCECITY' : 'COMMERCE CITY',
                 'DOCTOR PHILIPS' : 'PHILIPS',
                 'FAIRFIELD.' : 'FAIRFIELD',
                 'FEDERALHEIGHTS' : 'FEDERAL HEIGHTS',
                 'HERKIMER NY' : 'HERKIMER',
                 'HIDE A WAY HILLS' : 'HIDE-A-WAY HILLS',
                 'HOLLYWOOD HILLS EAST' : 'LOS ANGELES',
                 'IJAMES CHURCH RD' : 'MOCKSVILLE',
                 'JAMAICA EST' : 'JAMAICA',
                 'LA MOCA' : 'LAREDO',
                 'LACY LAKEVIEW' : 'LACY',
                 'LOS FELIZ L' : 'LOS ANGELES',
                 'MC ALLEN' : 'MCALLEN',
                 'MC LEANSVILLE' : 'MCLEANSVILLE',
                 'MT WASHINGTON' : 'PITTSBURGH',
                 'N MIAMI BEACH' : 'MIAMI BEACH',
                 'OLD BROOKLYN' : 'BROOKLYN',
                 'OLD MILL BASIN' : 'MILL BASIN',
                 'PARK HILLS HEIGHTS' : 'LOS ANGELES',
                 'POINT CHARLOTTE' : 'PORT CHARLOTTE',
                 'ST. ALBANS CITY' : 'SAINT ALBANS',
                 'SUPERIOR TWP' : 'ANN ARBOR',
                 'LYON TOWNSHIP' : 'SOUTH LYON',
                 'CLEAR LK SHRS' : 'CLEAR LAKE SHORES',
                 'BISCAYNE' : 'KEY BISCAYNE',
                 'SAINT ALBANS TOWN' : 'SAINT ALBANS',
                 'SODDY DAISY' : 'SODDY-DAISY'}
                             
    df_for_replace = df_for_replace.replace({'city_new' : cities_allias})                 
    result = pd.concat([df_for_replace, df_not_replace])
    
    return result

def main_get_cities(df):
    
# первая попытка - ищем по исходным названиям городов
    df[['geonameid','full_name', 'population', 'latitude', 'longitude']] = df.apply(get_city_full, attempt = 1, axis = 1)
    
# вторая попытка - ищем по синонимам, если не идентифицировали в первой попытке
    df = get_cities_allias(df)
    df[['geonameid','full_name', 'population', 'latitude', 'longitude']]  = df.apply(get_city_full, attempt = 2, axis = 1)
    
# третья попытка - выбираем населенный пункт с максимальной численностью, если на предыдущих попытках находилось несколько  
    df[['geonameid','full_name', 'population', 'latitude', 'longitude']] = df.apply(get_city_full, attempt = 3, axis = 1)

    return df



# 3. Для очистки признака 'schools'--------------------------------------------------------------

def clean_schools(df):
    df['schools_rating'] = df['schools_rating'].apply(lambda x: [item.replace('/10', '') for item in x])
    df['schools_rating'] = df['schools_rating'].apply(lambda x: [item.replace('NA', '') for item in x])
    df['schools_rating'] = df['schools_rating'].apply(lambda x: [item.replace('NR', '') for item in x])
    df['schools_rating'] = df['schools_rating'].apply(lambda x: [item.replace('None', '') for item in x])
    df['schools_rating'] = df['schools_rating'].apply(lambda x: list(filter(None, x)))
    df['schools_rating'] = df['schools_rating'].apply(lambda x: [int(elem) for elem in x])

    df['schools_grades'] = df['schools_grades'].apply(lambda x: [item.upper() for item in x if type(item) is str])
    df['schools_grades'] = df['schools_grades'].apply(lambda x: [item.replace(' TO ', '-') for item in x if type(item) is str])
    df['schools_grades'] = df['schools_grades'].apply(lambda x: [item.replace('–', '-') for item in x if type(item) is str])    
    df['schools_grades'] = df['schools_grades'].apply(lambda x: [item.replace('PRESCHOOL', 'K') for item in x if type(item) is str])
    df['schools_grades'] = df['schools_grades'].apply(lambda x: [item.replace('K–K', 'K') for item in x if type(item) is str])
    df['schools_grades'] = df['schools_grades'].apply(lambda x: [item.replace('PK–PK', 'PK') for item in x if type(item) is str])
    df['schools_grades'] = df['schools_grades'].apply(lambda x: [item.replace('NA', '') for item in x if type(item) is str])    
    df['schools_grades'] = df['schools_grades'].apply(lambda x: [item.replace('N/A', '') for item in x if type(item) is str])
    df['schools_grades'] = df['schools_grades'].apply(lambda lst: [y for x in lst for y in ([x] if ', ' not in x else [x.split(', ')[0],x.split(', ')[1]])])
    
    df['schools_distance'] = df['schools_distance'].apply(lambda x: [item.replace('mi', '') for item in x if type(item) is str])
    df['schools_distance'] = df['schools_distance'].apply(lambda x: [item.replace(' ', '') for item in x if type(item) is str])
    df['schools_distance'] = df['schools_distance'].apply(lambda x: list(filter(None, x)))
    df['schools_distance'] = df['schools_distance'].apply(lambda x: [float(elem) for elem in x])
    
    return df

def get_schools_grades_cover(schools_grades):
    
# список всех ступеней обучения
    all_grades_search = ['PK','K','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
    all_grades_found = ['PK','K','1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
    
# переменные для подсчета степени покрытия ступеней обучения
    add_cover = 1/15
    cover = 0
    
# далее ищем, какие ступени встречаются. В датасете могут быть указаны значения через запятую, через дефис (что означает
# включение всех ступеней между указанными крайними точками)
    for grade in all_grades_search:
        found = False
        for value in schools_grades:
            value = str(value)
            if '-' in value or '–' in value:
                lower_grade = value.split('-')[0]
                upper_grade = value.split('-')[1]
                if lower_grade == 'PK': start_slice = 0
                if lower_grade == 'K': start_slice = 1
                if upper_grade == 'K': stop_slice = 1
                if not lower_grade in ['PK','K']: start_slice = int(lower_grade)+1
                if not upper_grade in ['PK','K']: stop_slice = int(upper_grade)+2
                slice_object = slice(start_slice, stop_slice)
                sliced_grades_found = all_grades_found[slice_object]

                if grade in sliced_grades_found: 
                    if found == False:
                        cover += add_cover
                        found = True
            else:
                if grade == value:
                    if found == False:                    
                        cover += add_cover
                        found = True                    
    return round(cover,2)                  

def get_schools_measures(df):
    df['schools_rating_avg'] = df['schools_rating'].apply(lambda x: sum(x) / len(x) if len(x)>0 else '')
    df['schools_rating_max'] = df['schools_rating'].apply(lambda x: max(x) if len(x)>0 else '')

    df['schools_distance_avg'] =  df['schools_distance'].apply(lambda x: sum(x) / len(x) if len(x)>0 else '')
    df['schools_distance_min'] =  df['schools_distance'].apply(lambda x: min(x) / len(x) if len(x)>0 else '')
    
    df['schools_grades_cover'] = df['schools_grades'].apply(get_schools_grades_cover)
    return df



# 4. для очистки признака 'beds'-----------------------------------------------------------------

def change_beds_type(x):
    try:
        result = float(x)
    except:
        result = ''
    return result

def clear_beds_count(df):
    df['beds_count'] = df['beds_count'].str.upper()
    df['beds_count'] = df['beds_count'].apply(lambda x: np.nan if 'ACR' in str(x) else x)
    df['beds_count'] = df['beds_count'].apply(lambda x: np.nan if 'SQFT' in str(x) else x)
    df['beds_count'] = df['beds_count'].apply(lambda x: np.nan if 'BAEDONREDFIN' in str(x) else x)
    df['beds_count'] = df['beds_count'].str.replace(' ','')
    df['beds_count'] = df['beds_count'].str.replace('BD','')
    df['beds_count'] = df['beds_count'].str.replace('BEDS','')
    df['beds_count'] = df['beds_count'].str.replace('1BATH,3,CABLETVAVAILABLE,DININGROOM,EAT-INKITCHEN,LIVINGROOM,RANGE/OVEN,REFRIGERATOR','3')
    df['beds_count'] = df['beds_count'].str.replace('1BATH,2BEDROOMS','2')
    df['beds_count'] = df['beds_count'].str.replace('1BATH,2BEDROOMS,LIVINGROOM,RANGE/OVEN,REFRIGERATOR','2')
    df['beds_count'] = df['beds_count'].str.replace('1BATH,2BEDROOMS,EAT-INKITCHEN,LIVINGROOM,RANGE/OVEN,REFRIGERATOR','2')
    df['beds_count'] = df['beds_count'].str.replace('3ORMOREBEDROOMS','3')
    df['beds_count'] = df['beds_count'].str.replace('1BATH,3ORMOREBEDROOMS,CABLETVAVAILABLE,DININGROOM,EAT-INKITCHEN,LIVINGROOM,RANGE/OVEN,REFRIGERATOR','3')
    df['beds_count'] = df['beds_count'].str.replace('2,LIVINGROOM,RANGE/OVEN,REFRIGERATOR','2')
    df['beds_count'] = df['beds_count'].str.replace('2,EAT-INKITCHEN,LIVINGROOM,RANGE/OVEN,REFRIGERATOR','2')
    df['beds_count'] = df['beds_count'].str.replace('1,3,CABLETVAVAILABLE,DININGROOM,EAT-INKITCHEN,LIVINGROOM,RANGE/OVEN,REFRIGERATOR','3')    
    df['beds_count'] = df['beds_count'].str.replace('3,DININGROOM,LIVINGROOM,RANGE/OVEN,REFRIGERATOR','3')
    df['beds_count'] = df['beds_count'].str.replace('2BATHS,3','3')
    df['beds_count'] = df['beds_count'].str.replace('1-2','2')
    df['beds_count'] = df['beds_count'].str.replace('2,CABLETVAVAILABLE,DININGROOM,EAT-INKITCHEN,LIVINGROOM','2')
    df['beds_count'] = df['beds_count'].str.replace('#BEDROOMS1STFLOOR', '')
    df['beds_count'] = df['beds_count'].str.replace('--', '')
    df['beds_count'] = df['beds_count'].str.replace('BATH', '')
    df['beds_count'] = df['beds_count'].str.replace('BATHS', '')
    df['beds_count'] = df['beds_count'].str.replace('S', '')
    df['beds_count'] = df['beds_count'].apply(change_beds_type)
    df['beds_count'] = df['beds_count'].replace('', np.nan, regex = True)
    df['beds_count'] = df['beds_count'].astype(float, errors = 'ignore')
   
    return df



# 5. для очистки признака 'baths'---------------------------------------------------------------

def clear_baths(df):
    df['baths'] = df['baths'].str.upper()
    df['baths'] = df['baths'].str.replace(' ', '')
    df['baths'] = df['baths'].str.replace('BATHS', '')
    df['baths'] = df['baths'].str.replace('BATHROOMS', '')
    df['baths'] = df['baths'].str.replace('BA', '')
    df['baths'] = df['baths'].str.replace(':', '')
    df['baths'] = df['baths'].replace('116/116/116', np.nan, regex = True)
    df['baths'] = df['baths'].replace('SQ.FT', np.nan, regex = True)
    df['baths'] = df['baths'].str.replace('2-1/2-1/1-1/1-1', '4')
    df['baths'] = df['baths'].str.replace('1/1-0/1-0/1-0', '1')
    df['baths'] = df['baths'].str.replace('1/1/1/1', '4')
    df['baths'] = df['baths'].replace('--', np.nan, regex = True)
    df['baths'] = df['baths'].replace('~', np.nan, regex = True)
    df['baths'] = df['baths'].str.replace('SEMIMOD', '1')
    df['baths'] = df['baths'].str.replace('1-2', '2')
    df['baths'] = df['baths'].str.replace('3-1/2-2', '3')
    df['baths'] = df['baths'].str.replace('1-0/1-0/1', '3')    
    df['baths'] = df['baths'].replace('—', np.nan, regex = True)
    df['baths'] = df['baths'].str.replace('0/0', '0') 
    df['baths'] = df['baths'].str.replace(',', '.')     
    df['baths'] = df['baths'].apply(lambda x: float(str(x)[:-1]) + 0.25 if '+' in str(x) else x)
    df['baths'] = df['baths'].astype(float, errors = 'ignore').round(2)
    return df



# 6. для очистки признака 'stories'-----------------------------------------------------------

def clear_stories(df):
    df['stories_clear'] = df['stories'].astype(str)
    df['stories_clear'] = df['stories_clear'].str.upper()
    df['stories_clear'] = df['stories_clear'].str.replace(' LEVELS','')
    df['stories_clear'] = df['stories_clear'].str.replace(' LEVEL','')
    df['stories_clear'] = df['stories_clear'].str.replace(' stories_clear','')
    df['stories_clear'] = df['stories_clear'].str.replace(' STORY','')
    df['stories_clear'] = df['stories_clear'].str.replace('ONE','1')
    df['stories_clear'] = df['stories_clear'].str.replace('TWO','2')
    df['stories_clear'] = df['stories_clear'].str.replace('THREE','3')
    df['stories_clear'] = df['stories_clear'].str.replace('BI','2')
    df['stories_clear'] = df['stories_clear'].str.replace('TRI','3')
    df['stories_clear'] = df['stories_clear'].str.replace('-LEVEL','3')
    df['stories_clear'] = df['stories_clear'].str.replace(' OR MORE','')
    df['stories_clear'] = df['stories_clear'].str.replace(' TOWNHOUSE','')
    df['stories_clear'] = df['stories_clear'].str.replace(' ','')
    df['stories_clear'] = df['stories_clear'].str.replace('SITEBUILT','')
    df['stories_clear'] = df['stories_clear'].str.replace('RANCH','')
    df['stories_clear'] = df['stories_clear'].str.replace('+','')
    df['stories_clear'] = df['stories_clear'].str.replace('+','')

    df['stories_clear'] = df['stories_clear'].apply(lambda x: ''.join(i for i in str(x) if not i.isalpha()))
    df['stories_clear'] = df['stories_clear'].str.replace('/','')
    df['stories_clear'] = df['stories_clear'].str.replace(' ','')
    df['stories_clear'] = df['stories_clear'].str.replace('....','')
    #df['stories_clear'] = df['stories_clear'].str.replace(',,','')
    df['stories_clear'] = df['stories_clear'].str.replace('3,,33','3')
    df['stories_clear'] = df['stories_clear'].str.replace('3,,','3')
    df['stories_clear'] = df['stories_clear'].str.replace('1-2,3-4','4')
    df['stories_clear'] = df['stories_clear'].str.replace(', ','')
    df['stories_clear'] = df['stories_clear'].str.replace('(','')
    df['stories_clear'] = df['stories_clear'].str.replace(')','')
    df['stories_clear'] = df['stories_clear'].str.replace('3-5','4')
    df['stories_clear'] = df['stories_clear'].str.replace('1.5,2','2')
    df['stories_clear'] = df['stories_clear'].str.replace('2,3','3')
    df['stories_clear'] = df['stories_clear'].str.replace('3,3','3')
    df['stories_clear'] = df['stories_clear'].str.replace('2,,','2')
    df['stories_clear'] = df['stories_clear'].str.replace(',2,','2')
    df['stories_clear'] = df['stories_clear'].str.replace('3-3','3')
    df['stories_clear'] = df['stories_clear'].str.replace('-2,3-3','3')
    df['stories_clear'] = df['stories_clear'].str.replace('1,1.5','1.5')
    df['stories_clear'] = df['stories_clear'].str.replace(',,2','1.5')
    df['stories_clear'] = df['stories_clear'].str.replace('1,-','1')
    df['stories_clear'] = df['stories_clear'].str.replace('1,','1')
    df['stories_clear'] = df['stories_clear'].str.replace('3-4','4')
    df['stories_clear'] = df['stories_clear'].str.replace('1,1','1')
    df['stories_clear'] = df['stories_clear'].str.replace('1-2','2')
    df['stories_clear'] = df['stories_clear'].str.replace('1.5,,33','1.5')
    df['stories_clear'] = df['stories_clear'].str.replace(',','')
    df['stories_clear'] = df['stories_clear'].str.replace('-','')
    df['stories_clear'] = df['stories_clear'].str.replace('-','')
    df['stories_clear'] = df['stories_clear'].str.replace('1 Level, 2 Level','2')
    df['stories_clear'] = df['stories_clear'].str.replace('1 Story, 2 Story','2')
    df['stories_clear'] = df['stories_clear'].str.replace('1, 1','1')
    df['stories_clear'] = df['stories_clear'].str.replace('1, 1','1')
    df['stories_clear'] = df['stories_clear'].replace('', np.nan, regex = True)
    df['stories_clear'] = df['stories_clear'].astype(float, errors = 'ignore').round(2)
    
    return df

def round_stories(x):
    if x!=x: return x
    full = math.floor(x)
    decimal = x % 1
    result = x
    if decimal not in (0.5, 0):
        if decimal>0.25 and decimal<0.75:
            result  = full + 0.5
        else:
            result = round(x,0)
    return result

def change_stories(stories, stories_clear):
    result = stories_clear
    
# словарь для замены тесктовых этажей на числовые
    change_dict = {'Bi-Level' : 2,
                  'Condo 5+ Stories': 5,
                  'One and One Half' : 1.5,
                  'One, Three Or More' : 3,
                  'One, Two': 2,
                  'One, Two, Multi/Split': 2,
                  'Site Built, Tri-Level': 3,
                  'Split Entry (Bi-Level)': 2,
                  'Split Foyer, Tri-Level': 3,
                  'Tri-Level': 3,
                  '1 Level, 2 Level': 2,
                  '1 Story, 2 Story': 2,
                  '2 Story or 2 Level': 2,
                  '1, 1': 1}
    if stories in change_dict:
        result = change_dict[stories]
    return result



# 7. для очистки признака 'fireplace'----------------------------------------------------------

def negative_fireplace(x):
    negative_mask = ['NO','0', 0,'NOT APPLICABLE', 'N/K', 'NO FIREPLACE', 'NONE']
    if (x in negative_mask):
        return 0
    else:
        return 1
    
def clear_fireplace(df):
    df['fireplace_clear'] = df['fireplace'].str.upper()
    df['fireplace_clear'] = df['fireplace_clear'].replace('', np.nan, regex = True)
    df['fireplace_clear'] = df['fireplace_clear'].fillna(0)
    df['fireplace_clear'] = df['fireplace_clear'].apply(negative_fireplace)
    
    return df



# 8. для очистки признака 'propertytype'-------------------------------------------------------

def clear_propertytypes(df):

# Создаем новые признаки
    df['object_type'] = df['propertyType'].str.upper()
    df['home_style'] = df['propertyType'].str.upper()
    df['property_stories'] = df['propertyType'].str.upper()
    df['home_type'] = df['propertyType'].str.upper()

# Создаем справочники:
# Типы объекта недвижимости
    object_is_land = ['LOT/LAND', 'LAND']
    
# Стили строения    
    styles = {'COLONIAL': 'COLONIAL_HOME','CONTEMPORARY': 'CONTEMPORARY','COTTAGE': 'COTTAGE','CRAFTSMAN': 'CRAFTSMAN',
             'GREEK': 'GREEK','FARMHOUSE': 'FARMHOUSE','FRENCH': 'FRENCH','MEDITER': 'MEDITERRANIAN','MIDCENT': 'MIDCENTURY',
             'RANCH': 'RANCH','TUDOR': 'TUDOR','VICTOR': 'VICTORIAN','QUEEN': 'VICTORIAN','TRADITIONAL': 'TRADITIONAL',
             'FLORIDA': 'FLORIDA','LOG': 'LOG','CAPE' : 'CAPE COD','MODERN': 'MODERN','HISTOR': 'HISTORICAL',
             'OLDER': 'HISTORICAL', 'ART D': 'ART DECO', 'EUROP': 'EUROPEAN', 'SPANI': 'SPANISH'}
# Типы строения    
    home_types = {'SINGLEFAMILY' : 'SINGLE-FAMILY','SINGLE-FAMILY' : 'SINGLE-FAMILY','DETACHED' : 'SINGLE-FAMILY',
              'TINY-HOME' : 'SINGLE-FAMILY','TOWNHOUSE' : 'TOWNHOUSE','COOP' : 'COOPERATIVE','CO-OP' : 'COOPERATIVE',
              'CONDO' : 'CONDOMINIUM','APART' : 'APARTMENT','LOT/LAND' : 'LAND','LAND' : 'LAND',
              'MULTI-FAMILY' : 'MULTI-FAMILY','MULTIFAMILY' : 'MULTI-FAMILY','MULTIPLE' : 'MULTI-FAMILY','RANCH' : 'RANCH',
              'FARM' : 'RANCH','MOBILE' : 'MOBILE','CONTEMP' : 'CONTEMPORARY','COTTAGE': 'COTTAGE','HIGHRISE' : 'COOPERATIVE',
              'TRANSIT' : 'MOBILE','BUNGALOW' : 'SINGLE-FAMILY'}

# Заполним тип объекта
    df['object_type'] = df['object_type'].apply(lambda x: 'LAND' if x in object_is_land else 'HOME')

# Функция для определения типа строения
    def find_home_types(x):
        result = 'NOT A TYPE'
        for key in home_types:
            if key in x.replace(' ',''):
                result = home_types[key]
                return result
        return result

# Заполним тип строения и подчистим его
    df['home_type'] = df['home_type'].astype(str).apply(find_home_types)
    df['home_type'] = df['home_type'].replace('COTTAGE', 'OTHER')
    df['home_type'] = df['home_type'].replace('APARTMENT', 'OTHER')

# Функция для определения стиля строения 
    def find_styles(x):
        result = 'NOT A STYLE'
        for key in styles:
            if key in x:
                result = styles[key]
                return result
        return result

# Заполним стиль строения
    df['home_style'] = df['home_style'].astype(str).apply(find_styles)

# Выполним серию преобразований текста для выявления этажности 
    df.loc[(df['property_stories'].str.contains('ONE STORY', regex=True, na=False)), 'property_stories'] = '1STORY_NNN'
    df.loc[(df['property_stories'].str.contains('TWO STORY', regex=True, na=False)), 'property_stories'] = '2STORY_NNN'
    df.loc[(df['property_stories'].str.contains('TRI-LEVEL', regex=True, na=False)), 'property_stories'] = '3STORY_NNN'
    df['property_stories'] = df['property_stories'].apply(lambda x:
                                                          x[:x.index('STORIES') + len('STORIES')] if 'STORIES' in str(x) else x)
    df['property_stories'] = df['property_stories'].apply(lambda x:
                                                              x[:x.index('STORY') + len('STORY')] if 'STORY' in str(x) else x)
    df['property_stories'] = df['property_stories'].apply(lambda x:
                                                              x[:x.index('FLOOR') + len('FLOOR')] if 'FLOOR' in str(x) else x)
    df['property_stories'] = df['property_stories'].apply(lambda x:
                                                              x[:x.index('LEVEL') + len('LEVEL')] if 'LEVEL' in str(x) else x)
    df['property_stories'] = df['property_stories'].str.replace('ATTACHED OR 1/2 DUPLEX','')
    df['property_stories'] = df['property_stories'].str.replace('+','')
    df['property_stories'] = df['property_stories'].str.replace('(','')
    df['property_stories'] = df['property_stories'].str.replace(')','')
    df['property_stories'] = df['property_stories'].str.replace('1 1/2 ','')
    df['property_stories'] = df['property_stories'].str.replace(' ','')
    df['property_stories'] = df['property_stories'].str.replace('LOW-RISE','')
    df['property_stories'] = df['property_stories'].str.replace('CONDOMINIUM','')
    df['property_stories'] = df['property_stories'].str.replace('SPLIT','')
    df['property_stories'] = df['property_stories'].str.replace('HIGH-RISE','')
    df['property_stories'] = df['property_stories'].str.replace('TRADITIONAL,','')
    df['property_stories'] = df['property_stories'].str.replace('TOWNHOUSE,','')
    df['property_stories'] = df['property_stories'].str.replace('RANCH,','')
    df['property_stories'] = df['property_stories'].str.replace('1-3STORIES','2STORY')
    df['property_stories'] = df['property_stories'].str.replace('4-7STORIES','5STORIES')

# Функция для поиска числа этажей после предварительной подготовки данных
    def find_prop_digits(x):
        if ('STOR' in x) or ('FLOOR' in x) or ('LEVEL' in x):
            stories = re.sub('[^\d\.]', '', x)
            try:
                stories = float(stories)
                return stories
            except:
                return 'NO STORIES'
            return 'NO STORIES'

# Заполним число этажей
    df['property_stories'] = df['property_stories'].astype(str).apply(find_prop_digits)
    df['property_stories'] = df['property_stories'].fillna('NO STORIES')

# Заменим в признаке 'stories_clear' этажность, если в исходном признаке она была не указана и мы нашли ее в 'propertiesType'
    df['stories_clear']  = df.apply(lambda x: x['property_stories'] 
                                    if ((x['property_stories']!='NO STORIES') &
                                        (x['stories']!=x['stories']))
                                    else x['stories_clear'], axis =1)
    df['stories_clear'] = df['stories_clear'].astype(float)

# Удалим ненужные столбцы
    df.drop(columns = ['property_stories','home_style'], axis = 1, inplace=True)

# Возвращаем датафрейм с новыми признаками
    return df



# 9. для очистки признака 'homefacts'-------------------------------------------------------------

def get_homefacts(df):

# Получение из признака списка списков
    df['homeFacts'] = df['homeFacts'].apply(lambda x: eval(x)['atAGlanceFacts'])

# Функция для распаковки признака
    def unpack_homefacts(x):
        result = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        for fact_dict in x:
            if 'Year built' in fact_dict.values(): result[0] = fact_dict['factValue']
            if 'Remodeled year' in fact_dict.values(): result[1] = fact_dict['factValue']
            if 'Heating' in fact_dict.values(): result[2] = fact_dict['factValue']
            if 'Cooling' in fact_dict.values(): result[3] = fact_dict['factValue']  
            if 'Parking' in fact_dict.values(): result[4] = fact_dict['factValue'] 
            if 'lotsize' in fact_dict.values(): result[5] = fact_dict['factValue']
            if 'Price/sqft' in fact_dict.values(): result[6] = fact_dict['factValue']

        return result

# Распакуем признак на субпризнаки
    df[['facts_year_build','facts_remodeled_year','facts_heating',
          'facts_cooling','facts_parking','facts_lotsize','facts_price_per_sqft']] = df['homeFacts'].apply(unpack_homefacts)    

# Функция для очистки года постройки
    def clear_build(x):
        try:
            result = int(x)
        except:
            result = np.nan
        return result

# Очистим год постройки
    df['facts_year_build'] = df['facts_year_build'].apply(clear_build)

# Оставим только записи с годом постройки >=1900 и <=2023
    df = df[((df['facts_year_build']>=1900) & (df['facts_year_build']<=2023)) | (df['facts_year_build'].isnull())]
    
# Вычислим возраст строения
    max_fact_year_build = df['facts_year_build'].max()
    df['home_age'] = max_fact_year_build - df['facts_year_build']
    
# Заполним пропуски в возрасте строения - ЗДЕСЬ СТАВИМ МЕДИАНУ ИЗ ДЕФОЛТНЫХ
    df['home_age'] = df['home_age'].fillna(common_medians['home_age'])
    df.loc[(df['object_type']=='LAND'), 'home_age'] = 0
    
# Очистим год реконструкции функцией clear_build
    df['facts_remodeled_year'] = df['facts_remodeled_year'].apply(clear_build)
    
# Функция для получения возраста реконструкции
    def get_remodeled_age(df):
        df['remodeled_age'] = max_fact_year_build - df['facts_remodeled_year']
        df.loc[((df['facts_remodeled_year']==0) | 
                (df['facts_remodeled_year'].isnull()) |
                (df['remodeled_age']>=df['home_age'])),
                  'remodeled_age'] = df['home_age']
        return df

# Вычислим возраст реконструкции:
    df = get_remodeled_age(df)
    
# Функция для получения системы отопления
    def get_heating(fact_heat, obj_type):
        if obj_type == 'LAND': return 'NO HEATING NEED'
        if fact_heat != fact_heat: return 'NO DATA'
        fact_heat = str(fact_heat)
        result = []
        heating = ['AIR','GAS','ELECTRIC','NO DATA']
        for val in heating:
            if val in fact_heat: result.append(val)
        if len(result)==0: return 'OTHER'
        if len(result)>1: return 'MULTI'
        return result[0]

# Вычисление системы отопления
    df['facts_heating'] = df['facts_heating'].str.upper()
    df['heating_system'] = df.apply(lambda x: get_heating(x['facts_heating'], x['object_type']), axis = 1)
    
# Функция для вычисления системы охлаждения
    def get_cooling(fact_cool, obj_type):
        if obj_type == 'LAND': return 'NO COOLING NEED'
        if fact_cool != fact_cool: return 'NO DATA'
        if fact_cool == 'NO DATA': return 'NO DATA'
        return 'HAS COOLING'

# Вычисление системы охлаждения
    df['facts_cooling'] = df['facts_cooling'].str.upper()
    df['cooling_system'] = df.apply(lambda x: get_cooling(x['facts_cooling'], x['object_type']), axis = 1)
    
# Функция для вычисления наличия и концепции парковки
    def get_parking(parking):
        if parking != parking: return 'NO DATA'
        parking=str(parking).replace('-',' ')
        garages = ['ATTACHED','DETACHED','OFF STREET', 'ON STREET','DRIVEWAY','CARPORT', 'NO DATA']
        result = []
        for val in garages:
            if val in parking: result.append(val)
        if len(result)==0: return 'OTHER'
        if len(result)>1: return 'MULTI'
        return result[0]

# Вычислим наличие и систему парковки
    df['facts_parking'] = df['facts_parking'].str.upper()
    df['parking_type'] = df['facts_parking'].apply(get_parking)

# Функция для первичной очистки площади объекта
    def clear_lotsize(df):
        df['lotsize'] = df['facts_lotsize'].str.upper()
        df['lotsize'] = df['lotsize'].str.replace(' ','')
        df['lotsize'] = df['lotsize'].str.replace('LOT','')
        df['lotsize'] = df['lotsize'].str.replace('SQ.FT.','SQFT')
        df['lotsize'] = df['lotsize'].str.replace('-','')
        df['lotsize'] = df['lotsize'].str.replace('—','') 
        df['lotsize'] = df['lotsize'].str.replace(',','')    
        return(df)

# Очистим площадь отъекта
    df = clear_lotsize(df)
    
# Функция для финальной очистки площади объекта
    def get_lotsize(lotsize):
        acr = 43560.0391
        if lotsize!=lotsize: return 'NO DATA'
        lotsize = str(lotsize).upper()
        measure = lotsize.replace('\d+', '')
        square = re.sub('[^\d\.]', '', lotsize)
        try:
            square = float(square)
        except:
            return 'NO DATA'
        if 'ACRE' in measure: square = square * acr
        return square

# Выполним финальную проверку площади объекта
    df['lotsize'] = df['lotsize'].apply(get_lotsize)
    
# Функция для перехода к признаку добавочной площади и корректировки признака 'sqft'
    def check_lotsize_sqft(lot, sqft, obj_type):
        if lot=='NO DATA': lot = 0
        lot = float(lot)
        sqft = float(sqft)
        if obj_type == 'LAND':
            if lot > sqft:
                result = pd.Series([0, lot])
                return result
            else:
                result = pd.Series([0, sqft])
                return result
        else:
            if lot > sqft:
                result = pd.Series([lot-sqft, sqft])
                return result
            else:
                result = pd.Series([0, sqft])
                return result
        result = pd.Series(['NO','NO'])
        return result

# Вычислим добавочную площадь и актуализированный 'sqft'
    df[['additional_sqft','sqft_clear']] = df.apply(lambda x: check_lotsize_sqft(x['lotsize'], x['sqft'], x['object_type']), axis=1)
    
# Удалим избыточные признаки

    colnames = ['facts_year_build', 'facts_remodeled_year', 'facts_heating', 'facts_cooling',
              'facts_parking', 'facts_lotsize','facts_price_per_sqft', 'lotsize', 'sqft', 'homeFacts']  
    
    df.drop(columns = colnames, axis = 1, inplace = True)

# Возвращаем актуализированный датасет
    return df



# 10. для определения расстояния до ближайшей столицы штата-------------------------------------------

def get_capital_distance(lat, long):
    distance = []
    
# если какой-то из координат нет - возвращаем пропуск
    if lat!= lat or long!=long: return np.nan
    lat1 = float(lat)
    long1 = float(long)

# вычисляем расстояние до каждой столицы штата и записываем в словарь
    for state in states_capitals:
        capital_coord = states_capitals[state].split(',')
        lat2 = float(capital_coord[0])
        long2 = float(capital_coord[1])
        dist = haversine([lat1,long1], [lat2,long2], unit='mi')
        distance.append(dist)
        
# возвращаем минимальное расстояние
    result = min(distance)
    
    return result


# 11. для заполнения пропусков по признакам 'schools'------------------------------------------------

def fill_schools_nan(df):
    
# присоединим к датафрейму медианы по городам и заполним пропуски присоединенными медианами
    df = df.merge(schools_medians_city, on = ['city','state'], how = 'left')
    df['schools_rating_avg_x'] = df['schools_rating_avg_x'].fillna(df['schools_rating_avg_y'])
    df['schools_rating_max_x'] = df['schools_rating_max_x'].fillna(df['schools_rating_max_y'])
    df['schools_distance_avg_x'] = df['schools_distance_avg_x'].fillna(df['schools_distance_avg_y'])
    df['schools_distance_min_x'] = df['schools_distance_min_x'].fillna(df['schools_distance_min_y'])
    df['schools_grades_cover_x'] = df['schools_grades_cover_x'].fillna(df['schools_grades_cover_y'])
    
# а если пропуски после этого остались, то заменим их на медианы из словаря (медианы по всем городам)  
    df['schools_rating_avg_x'] = df['schools_rating_avg_x'].fillna(schools_medians_all['schools_rating_avg'])
    df['schools_rating_max_x'] = df['schools_rating_max_x'].fillna(schools_medians_all['schools_rating_max'])
    df['schools_distance_avg_x'] = df['schools_distance_avg_x'].fillna(schools_medians_all['schools_distance_avg'])
    df['schools_distance_min_x'] = df['schools_distance_min_x'].fillna(schools_medians_all['schools_distance_min'])
    df['schools_grades_cover_x'] = df['schools_grades_cover_x'].fillna(schools_medians_all['schools_grades_cover']) 
    
    
# удалим лишние образовавшиеся признаки и переименуем    
    df.drop(['schools_rating_avg_y','schools_rating_max_y','schools_distance_avg_y','schools_distance_min_y',
             'schools_grades_cover_y'], inplace = True, axis=1)
    df = df.rename(columns={'schools_rating_avg_x': 'schools_rating_avg',
                            'schools_rating_max_x' : 'schools_rating_max',
                            'schools_distance_avg_x': 'schools_distance_avg',
                           'schools_distance_min_x': 'schools_distance_min',
                            'schools_grades_cover_x': 'schools_grades_cover'})
    
    return df



# 12. охватывающая функция для выполнения всех преобразований признаков-----------------------------
# не будем еще раз расписывать комментарии, это все преобразования, выполнявшиеся ранее

def clear_input(df):
    
# 12.0. Обработка признака 'status'

    def normalize_status(df):
        df['status'] = df['status'].str.lower()
        df['status'] = df['status'].apply(normalise_status)
        
        return df

# 12.1. Обработка признаков 'private pool'

    def normalize_pool(df):
        df['PrivatePool'] = df['PrivatePool'].str.lower()
        df['private pool'] = df.apply(normalise_pool, axis = 1)
        df['private pool'] = df['private pool'].apply(lambda x: 1 if x=='Yes' else 0)
        df['private pool'] = df['private pool'].astype(int)
        df = df.drop('PrivatePool', axis = 1)

        return df

# 12.2. Обработка признака 'status'

    def normalize_status(df):
        df['status'] = df['status'].str.lower()
        df['status'] = df['status'].apply(normalise_status)
        
        return df

# 12.3. Обработка признака 'city'
  
    def normalize_city(df):
# создаем объект geonamescache
        gc = geonamescache.GeonamesCache(min_city_population=500)
        df['city'] = df['city'].str.upper()

# собираем список уникальных городов из датафрейма
        grouped_city = df.groupby(['city','state']).size().reset_index()
        grouped_city = grouped_city.rename(columns={0: 'count'})

# прогоняем список уникальных городов через функции для их идентификации
        grouped_city[['geonameid','name','population','latitude','longitude']] = grouped_city.apply(get_city, axis = 1)
        grouped_city = main_get_cities(grouped_city)
        
# присоединяем к датафрейму признаки по городам
        df = df.merge(grouped_city,
              on = ['city','state'],
              how = 'left')
    
# наводим красоту в оформлении
        df.drop(['city','name','city_new','count','geonameid'], inplace = True, axis=1)
        df = df.rename(columns={'full_name': 'city'})
        
# заполнеяем пропуски
        no_city = ['OTHER CITY-1', 'Empty input']
        df['city'] = df['city'].apply(lambda x: np.nan if str(x) in no_city else x)
        df['population'] = df['population'].fillna(common_medians['population'])
        df['nearest_capital'] = df.apply(lambda x: get_capital_distance(x['latitude'], x['longitude']), axis = 1)
        df.loc[(df['city'].isnull()), 'nearest_capital'] = common_medians['distance']
        
# удаляем более не нужные признаки координат
        df.drop(['latitude','longitude'], inplace = True, axis = 1)
        
        return df

# 12.4. Обработка признака 'State'

    def normalize_state(df):
        df['state'] = df['state'].str.upper()
        df['state'] = df['state'].apply(lambda x: 'OTHER' if x in other_states else x)
        
        return df

# 12.5. Обработка признака 'schools'

    def normalize_schools(df):
        
# подготавливаем текст для распаковки
        df['schools'] = df['schools'].str.replace('\\','')
        df['schools'] = df['schools'].apply(lambda x: x[1:])     
        df['schools'] = df['schools'].apply(lambda x: x[:-1]) 
        
# распаковываем признак в новые субпризнаки
        df['schools_rating'] = df['schools'].apply(lambda x: eval(x)['rating'])
        df['schools_distance'] = df['schools'].apply(lambda x: eval(x)['data']['Distance'])
        df['schools_grades'] = df['schools'].apply(lambda x: eval(x)['data']['Grades'])

# применяем функции очистки и преобразования к новым субпризнакам
        df = clean_schools(df)
        df = get_schools_measures(df)
        df = fill_schools_nan(df)
        
# удаляем лишние столбцы
        df.drop(['schools','schools_rating','schools_distance','schools_grades'], axis = 1, inplace = True)
    
        return df

# 12.6. Обработка признака 'sqft'

    def normalize_sqft(df):
        df['sqft'] = df['sqft'].astype(str)
        df['sqft'] = df['sqft'].apply(lambda x: re.sub( r'[^0-9]+', '', x))
        df['sqft'] = df['sqft'].replace('', np.nan, regex = True)
        df['sqft_clear'] = df['sqft'].fillna(common_medians['sqft'])
        df['sqft_clear'] = df['sqft_clear'].astype(float)
        
        return df

# 12.7. Обработка признака 'stories'
    
    def normalize_stories(df):
        
# применяем основные функции очитки
        df = clear_stories(df)
        df['stories_clear'] = df['stories_clear'].apply(round_stories)

# заменяем некоторые аномальные значения
        df['stories_clear'] = df['stories_clear'].apply(lambda x: 1.5 if x in [112,113] else x)
        df['stories_clear'] = df.apply(lambda x: change_stories(x['stories'], x['stories_clear']), axis = 1)
        df.loc[(df['propertyType']=='lot/land') | (df['propertyType']=='Land'), 'stories_clear'] = 0

        df['sqft_per_story'] = df['sqft_clear'] / df['stories_clear']
        df.loc[((df['sqft_per_story']<600) & (df['stories_clear']>4)), 'stories_clear'] = 1

        df.loc[((df['propertyType']!='lot/land') &
                 (df['propertyType']!='Land')) &
                 (df['stories_clear']==0),'stories_clear'] = 1

# меняем пропуски на 1
        df['stories_clear'].fillna(1, inplace=True)
        
        return df

# 12.8. Обработка признака 'beds'

    def normalize_beds(df):
        df['beds_count'] = df['beds']
                                       
# применяем основную функцию очистки
        df = clear_beds_count(df)
        df.loc[(df['propertyType']=='lot/land') | (df['propertyType']=='Land'), 'beds_count'] = 0
        df['beds_count'] = df['beds_count'].astype(float)
    
        return df

# 12.9. Обработка признака 'baths'

    def normalize_baths(df):

# применяем основную функцию очистки
        df = clear_baths(df)
                                       
# заполняем пропуски и аномальные значения
        df['baths'] = np.where(df['baths']>(df['beds_count']+1+df['private pool'])*2, df['beds_count'], df['baths'])
        df.loc[(df['propertyType']=='lot/land') | (df['propertyType']=='Land'), 'baths'] = 0
        df['baths'] = np.where((df['baths'].isnull()) &
                                 (df['propertyType']!='lot/land') &
                                 (df['propertyType']!='Land'), (df['beds_count']+df['private pool']), df['baths'])
    
        return df

# 12.10. Обработка признака 'fireplace'

    def normalize_fireplace(df):
                                       
# применяем основную функцию очистки
        df = clear_fireplace(df)
                                       
# заполняем пропуски и меняем аномальные значения
        df[((df['propertyType']=='Land') | (df['propertyType']=='lot/land')) & df['fireplace_clear']==1].shape[0]
        df.loc[((df['propertyType']!='lot/land') &
                 (df['propertyType']!='Land')) &
                 (df['fireplace_clear']==1), 'fireplace_clear'] = 0
    
        return df

# 12.11. Обработка признака 'propertytypes'

    def normalize_propertytypes(df):
                                       
# применяем основную функцию проверки
        df = clear_propertytypes(df)
                                       
# заполняем пропуски и меняем аномальные значения
        df['sqft'] = df['sqft'].fillna(df['home_type'].map(hometype_medians_sqft))
        df['sqft'] = df.apply(lambda x: hometype_medians_sqft[x['home_type']] if (x['sqft']==0) else x['sqft'], axis=1)
        df['sqft_clear'] = df['sqft']
        df['sqft_clear'] = df['sqft_clear'].astype(float)     
    
        return df

# 12.12. Обработка признака 'homefacts'

    def normalize_homefacts(df):
                                       
# применяем основную функцию очистки и преобразования
        df = get_homefacts(df)
                                       
# заполняем пропуски и меняем аномальные значения
        df['sqft_per_bed'] = df['sqft_clear'] / df['beds_count']
        df['beds_count'] = df['beds_count'].fillna(round(df['sqft_clear'] / common_medians['sqft_per_bed'], 0))
        df['beds_count'] = df['beds_count'].apply(lambda x: 100 if x>100 else x)
        df['baths'] = df['baths'].fillna(df['beds_count'] + df['private pool'])
        
        return df

# 12.13. Обработка признака 'target', если его подали в новых данных

    def normalize_target(df):
                                       
# если признак подали в сервис, то выполняем очистку по ранее описанному алгоритму
        if 'target' in df.columns:
            df['target'] = df['target'].str.replace(r"[^\d\.]","", regex=True)
            df['target'] = df['target'].astype(float)
                                       
# разделяем признак на 2 массива - для основных (продажа) сделок и второстепенных (аренда)
            target_main = df[df['status']!='for rent']['target']
            target_rent = df[df['status']=='for rent']['target']        
        return df, target_main, target_rent

# 12.14. Оставляем в наборе только чистые признаки
    
    def leave_clear(df):
        features = ['status','private pool','state', 'schools_rating_max', 'schools_rating_avg',
                    'schools_distance_avg','schools_distance_min','schools_grades_cover', 'beds_count',
                    'stories_clear','fireplace_clear','object_type','home_type', 'home_age','remodeled_age',
                    'heating_system','cooling_system','parking_type','additional_sqft','sqft_clear','population', 'nearest_capital']
        df = df[features]
    
        return df

# 12.15. Разделение входящего датасета на сделки по продаже и аренде для маршрутизации в разные модели

    def separate_data(df):
        df_main = df[df['status']!='for rent']
        df_rent = df[df['status']=='for rent']
        return df_main, df_rent

# 12.16. Применение OheHotEncoder

    def dummies(df):
                                       
# только если подан не пустой датафрейм
        if df.empty==False:
            df = pd.get_dummies(df)
        
        return df    

# 12.17. Удаление признаков с высокой корреляцией в датасете по продаже
    
    def leave_corr_main(df):
                                       
# удаление делаем циклом, проверяя, что удаляемый признак вообще есть в датафрейме
        for feature in features_for_del:
            if feature in df.columns:
                df.drop(feature, axis=1, inplace = True)
    
        return df
    
# 12.18. Удаление признаков с высокой корреляцией в датасете по аренде
    
    def leave_corr_rent(df):
                                       
# удаление делаем циклом, проверяя, что удаляемый признак вообще есть в датафрейме
        for feature in rent_features_for_del:
            if feature in df.columns:
                df.drop(feature, axis=1, inplace = True)
        return df

# 12.19. Добавление пустых признаков для модели по продаже

    def add_features_main(df):
                                       
# проверяем поочередно наличие каждого из признаков, ожидаемых моделью, в датасете; если нет - добавляем со значением 0
        for feature in model_features:
            if not feature in df.columns:
                df[feature] = 0
                df[feature] = df[feature].astype(float)
        df = df.reindex(columns=model_features)
        
        return df
    
# 12.20. Добавление пустых признаков для модели по аренде

    def add_features_rent(df):
                                       
# проверяем поочередно наличие каждого из признаков, ожидаемых моделью, в датасете; если нет - добавляем со значением 0
        for feature in rent_model_features:
            if not feature in df.columns:
                df[feature] = 0
                df[feature] = df[feature].astype(float)
        df = df.reindex(columns=rent_model_features)
        
        return df
    
# 12.21. Стандартизация признаков для модели по продаже

    def standard_main(df):
                                       
# выполняем, только если датасет признаков по аренде вообще есть
        if df.empty==False:
            cols = df.columns
            df = main_scaler.transform(df)
            df = pd.DataFrame(df, columns = cols)
        return df
    
# 12.22. Стандартизация признаков для модели по аренде

    def standard_rent(df):
                                       
# выполняем, только если датасет признаков по аренде вообще есть
        if df.empty==False:
            cols = df.columns
            df = rent_scaler.transform(df)
            df = pd.DataFrame(df, columns = cols)
        return df

    
# 13. Применяем преобразования - последовательно выполняем перечисленные выше функции
    df = normalize_status(df)
    df = normalize_pool(df)
    df = normalize_status(df)
    df = normalize_city(df)
    df = normalize_state(df)
    df = normalize_schools(df)
    df = normalize_sqft(df)
    df = normalize_stories(df)
    df = normalize_beds(df)
    df = normalize_baths(df)
    df = normalize_fireplace(df)
    df = normalize_propertytypes(df)
    df = normalize_homefacts(df)
    df, target_main, target_rent = normalize_target(df)
    df = leave_clear(df)
    df_main, df_rent = separate_data(df)
    df_main = dummies(df_main)
    df_rent = dummies(df_rent)
    df_main = leave_corr_main(df_main)
    df_rent = leave_corr_rent(df_rent)
    df_main = add_features_main(df_main)
    df_rent = add_features_rent(df_rent)
    df_rent = df_rent[rent_model_features]
    df_main = standard_main(df_main)    
    df_rent = standard_rent(df_rent)
    return df_main, df_rent, target_main, target_rent

# СЕРВИСНАЯ ЧАСТЬ
try:
    # Создаём подключение по адресу rabbitmq
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbitmq'))
    channel = connection.channel()
    # Объявляем очередь input
    channel.queue_declare(queue='input')
    # Создаём функцию callback для обработки данных из очереди
    def callback_input(ch, method, properties, body):
        # Запишем событие в лог
        log_string = str( datetime.datetime.now()) + ' Features: Поступили новые данные для трансформации для модели'
        with open('./logs/service_log.txt', 'a', encoding='utf-8') as log:
                log.write(log_string +'\n')
        # десериализируем полученные данные
        msg = json.loads(body)
        # переведем обратно в датафрейм
        df_input = pd.DataFrame(msg['data'], columns = msg['columns'])    
        # выполним подготовку данных и разделение на данные для разных моделей
        # и целевые признаки, если они есть в исходных данных                
        df_main, df_rent, target_main, target_rent = clear_input(df_input)

        # если есть признаки для сделок продажи - передаем их в модель
        if df_main.empty==False:
            # сериализируем данные для сообщения
            df_main = df_main.to_json(orient='split') 
            # отправляем в очередь     
            channel.basic_publish(exchange='',
                            routing_key='features_main',
                            body=df_main)
             # Запишем событие в лог
            log_string = str( datetime.datetime.now()) + ' Features: Признаки для расчета цен продажи отправлены в очередь features_main'
            with open('./logs/service_log.txt', 'a', encoding='utf-8') as log:
                log.write(log_string +'\n')
        # если есть признаки для сделок аренды - передаем их в модель
        if df_rent.empty==False:
            # сериализируем данные для сообщения
            df_rent = df_rent.to_json(orient='split') 
            # отправляем в очередь   
            channel.basic_publish(exchange='',
                            routing_key='features_rent',
                            body=df_rent)  
            log_string = str( datetime.datetime.now()) +  ' Features: Признаки для расчета цен аренды отправлены в очередь features_rent'
            with open('./logs/service_log.txt', 'a', encoding='utf-8') as log:
                log.write(log_string +'\n')
        # если на входе есть цены продажи - передаем их в очередь просто информационно
        if target_main.empty==False:
            # сериализируем данные
            target_main = target_main.to_json(orient='split')   
            # отправляем в очередь
            channel.basic_publish(exchange='',
                            routing_key='target_main',
                            body=target_main)
            # Запишем событие в лог
            log_string = str( datetime.datetime.now()) + ' Features: Справочно: из поступивших данных текущие цены продажи отправлены очередь target_main'
            with open('./logs/service_log.txt', 'a', encoding='utf-8') as log:
                log.write(log_string +'\n')
        # если на входе есть цены аренды - передаем их в очередь просто информационно
        if target_rent.empty==False:
            # сериализируем данные
            target_rent = target_rent.to_json(orient='split') 
            # отправляем в очередь   
            channel.basic_publish(exchange='',
                            routing_key='target_rent',
                            body=target_rent)       
            # Запишем событие в лог
            log_string = str( datetime.datetime.now()) + ' Features: Справочно: из поступивших данных текущие цены аренды отправлены очередь target_rent'
            with open('./logs/service_log.txt', 'a', encoding='utf-8') as log:
                log.write(log_string +'\n')
    # Извлекаем сообщение из очередей признаков для моделей
    # on_message_callback показывает, какую функцию вызвать при получении сообщения
    channel.basic_consume(
        queue='input',
        on_message_callback=callback_input,
        auto_ack=True
    )
    # Запускаем режим ожидания прихода сообщений
    print('...Ожидание сообщений, для выхода нажмите CTRL+C')
 
    # Запускаем режим ожидания прихода сообщений
    channel.start_consuming()

except:
   print('Не удалось подключиться к очереди (features-сервис)')