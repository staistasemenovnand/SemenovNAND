import pika
import json
import pickle
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
import datetime
from sklearn.linear_model import LinearRegression

#----------------- ЗАГРУЗКА СОХРАНЕННЫХ МОДЕЛЕЙ--------------------------------------------------

with open('RFR_model.pickle', 'rb') as mm:
    main_model = pickle.load(mm)
    
with open('LR_OLS_model.pickle', "rb") as rm:
    rent_model = pickle.load(rm)

# Попытка работы сервера
try:
    # Создаём подключение по адресу rabbitmq
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbitmq'))
    channel = connection.channel()
 
    # Объявляем очередь features_main
    channel.queue_declare(queue='features_main')
    # Объявляем очередь features_rent
    channel.queue_declare(queue='features_rent')
    # Объявляем очередь price_main
    channel.queue_declare(queue='price_main')
    # Объявляем очередь price_rent
    channel.queue_declare(queue='price_rent')
 
    # Создаём функцию для обработки данных из очереди признаков для продажи
    def callback_main(ch, method, properties, body):
        # десериализируем данные из сообщения
        msg = json.loads(body)
        # переведем обратно в датафрейм
        df_main = pd.DataFrame(msg['data'], columns = msg['columns'])                    

        # если датафрейм признаков для продажи не пустой - передаем его в модель
        if df_main.empty==False:
        # Запишем событие в лог
            log_string =str( datetime.datetime.now()) +' Model: В очередь features_main поступили новые данные для расчета цен продажи'
            with open('./logs/service_log.txt', 'a', encoding='utf-8') as log:
                log.write(log_string +'\n')
            # выполним предсказания
            price_main = main_model.predict(df_main)
            # переведем из логарифма в USD
            price_main = np.exp(price_main)
            # сериализируем для передачи в результирующую очередь
            price_main = list(price_main)   
            price_main = json.dumps(price_main)
            # опубликуем предсказанные цены продажи в результирующей очереди     
            channel.basic_publish(exchange='',
                            routing_key='price_main',
                            body=price_main)
            print(price_main)
            # Запишем событие в лог
            log_string = str( datetime.datetime.now()) + ' Model: В очередь price_main отправлены прогнозные цены продажи:' + '\n' + price_main
            with open('./logs/service_log.txt', 'a', encoding='utf-8') as log:
                log.write(log_string +'\n')

    # Создаём функцию для обработки данных из очереди признаков для продажи
    def callback_rent(ch, method, properties, body):
        # десериализируем данные из сообщения
        msg = json.loads(body)
        # переведем обратно в датафрейм
        df_rent = pd.DataFrame(msg['data'], columns = msg['columns'])                    

        # если датафрейм признаков для продажи не пустой - передаем его в модель
        if df_rent.empty==False:
        # Запишем событие в лог
            log_string = str( datetime.datetime.now()) + ' Model: В очередь features_rent поступили новые данные для расчета цен аренды'
            with open('./logs/service_log.txt', 'a', encoding='utf-8') as log:
                log.write(log_string +'\n')
            # выполним предсказания
            price_rent = rent_model.predict(df_rent)
            # переведем из логарифма в USD
            price_rent = np.exp(price_rent)
            # сериализируем для передачи в результирующую очередь
            price_rent = list(price_rent)   
            price_rent = json.dumps(price_rent)
            # опубликуем предсказанные цены аренды в результирующей очереди              
            channel.basic_publish(exchange='',
                            routing_key='price_rent',
                            body=price_rent)
            print(price_rent)
            # Запишем событие в лог
            log_string = str( datetime.datetime.now()) + ' Model: В очередь price_rent отправлены прогнозные цены аренды:' + '\n' + price_rent
            with open('./logs/service_log.txt', 'a', encoding='utf-8') as log:
                log.write(log_string +'\n')

    # Извлекаем сообщение из очередей признаков для моделей
    # on_message_callback показывает, какую функцию вызвать при получении сообщения
    channel.basic_consume(
        queue='features_main',
        on_message_callback=callback_main,
        auto_ack=True
    )
    channel.basic_consume(
        queue='features_rent',
        on_message_callback=callback_rent,
        auto_ack=True
    )
    print('...Ожидание сообщений, для выхода нажмите CTRL+C')
 
    # Запускаем режим ожидания прихода сообщений
    channel.start_consuming()
except:
    print('Не удалось подключиться к очереди (model-сервис)')


