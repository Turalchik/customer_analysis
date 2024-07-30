import streamlit as st
import pandas as pd
from model import *
from PIL import Image


def process_main_page():
    show_main_page()
    process_side_bar_inputs()


def show_main_page():
    image = Image.open('data/mouse.jpg')
    max_size = (800, 600)
    image.thumbnail(max_size)

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Demo Clients",
        page_icon=image,

    )

    st.write(
        """
        # Классификация клиентов авиакомпании
        Определяем кто будет доволен услугами, а кто нет

        ###### вот тут мики-мики
        """
    )

    st.image(image)


def write_user_data(df):
    st.write("## Ваши данные")
    st.write(df)


def write_prediction(prediction, prediction_probas):
    st.write("## Предсказание")
    st.write(prediction)

    st.write("## Вероятность предсказания")
    st.write(prediction_probas)


def process_side_bar_inputs():
    st.sidebar.header('Заданные пользователем параметры')
    user_input_df = sidebar_input_features()

    user_X_df = process_data(user_input_df)
    write_user_data(user_X_df)

    prediction, prediction_probas = load_model_and_predict(user_X_df)
    write_prediction(prediction, prediction_probas)


def sidebar_input_features():

    gender = st.sidebar.selectbox("Пол", ("Мужской", "Женский", "Другой"))
    customer_type = st.sidebar.selectbox("Тип клиента", ("Постоянный клиент", "Нелояльный клиент", "Не понятно"))
    type_of_travel = st.sidebar.selectbox("Цель поездки", ("По делам", "Просто так", "Сложно сказать"))
    class_travel = st.sidebar.selectbox("Класс", ("Бизнес", "Эконом +", "Эконом", "Я не помню"))

    age = st.sidebar.slider("Возраст", min_value=0, max_value=80, value=20, step=1)
    flight_distance = st.sidebar.slider("Расстояние полета", min_value=1, max_value=10000, value=1000, step=1)
    departure_delay_in_minutes = st.sidebar.slider("Задержка отправления на несколько минут", min_value=1, max_value=100, value=50, step=1)
    arrival_delay_in_minutes = st.sidebar.slider("Задержка прибытия на несколько минут", min_value=1, max_value=100, value=50, step=1)
    inflight_wifi_service = st.sidebar.slider("Услуга Wi-Fi в полете", min_value=1, max_value=100, value=50, step=1)
    departure_arrival_time_convenient = st.sidebar.slider("Удобное время отправления и прибытия", min_value=1, max_value=100, value=50, step=1)
    ease_of_online_booking = st.sidebar.slider("Простота онлайн-брониро", min_value=1, max_value=100, value=50, step=1)
    gate_location = st.sidebar.slider("Местоположение выхода на посадку", min_value=1, max_value=100, value=50, step=1)
    food_and_drink = st.sidebar.slider("Еда и напитки", min_value=1, max_value=100, value=50, step=1)
    online_boarding = st.sidebar.slider("Онлайн-посадка на борт", min_value=1, max_value=100, value=50, step=1)
    seat_comfort = st.sidebar.slider("Комфорт сиденья", min_value=1, max_value=100, value=50, step=1)
    inflight_entertaiment = st.sidebar.slider("Развлечение в полете", min_value=1, max_value=100, value=50, step=1)
    on_board_service = st.sidebar.slider("Бортовой сервис", min_value=1, max_value=100, value=50, step=1)
    leg_room_service = st.sidebar.slider("Дополнительное пространство для ног", min_value=1, max_value=100, value=50, step=1)
    baggage_handling = st.sidebar.slider("Обработка багажа", min_value=1, max_value=100, value=50, step=1)
    checkin_service = st.sidebar.slider("Служба проверки", min_value=1, max_value=100, value=50, step=1)
    inflight_service = st.sidebar.slider("Обслуживание в полете", min_value=1, max_value=100, value=50, step=1)
    cleanliness = st.sidebar.slider("Чистота", min_value=1, max_value=100, value=50, step=1)

    translation = {
        'Мужской': 'Male',
        'Женский': 'Female',
        'Другой': 'unknown',
        'Постоянный клиент': 'Loyal Customer',
        'Нелояльный клиент': 'disloyal Customer',
        'Не понятно': 'unknown',
        'По делам': 'Business travel',
        'Просто так': 'Personal travel',
        'Сложно сказать': 'unknown',
        'Бизнес': 'Business',
        'Эконом +': 'Eco Plus',
        'Эконом': 'Eco',
        'Я не помню': 'unknown'
    }

    data = {
        'Gender': translation[gender],
        'Age': age,
        'Customer Type': translation[customer_type],
        'Type of Travel': translation[type_of_travel],
        'Class': translation[class_travel],
        'Flight Distance': flight_distance,
        'Departure Delay in Minutes': departure_delay_in_minutes,
        'Arrival Delay in Minutes': arrival_delay_in_minutes,
        'Inflight wifi service': inflight_wifi_service,
        'Departure/Arrival time convenient': departure_arrival_time_convenient,
        'Ease of Online booking': ease_of_online_booking,
        'Gate location': gate_location,
        'Food and drink': food_and_drink,
        'Online boarding': online_boarding,
        'Seat comfort': seat_comfort,
        'Inflight entertainment': inflight_entertaiment,
        'On-board service': on_board_service,
        'Leg room service': leg_room_service,
        'Baggage handling': baggage_handling,
        'Checkin service': checkin_service,
        'Inflight service': inflight_service,
        'Cleanliness': cleanliness
    }

    df = pd.DataFrame(data, index=[0])

    return df


if __name__ == "__main__":
    process_main_page()
