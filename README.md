
# Классификация клиентов авиакомпании

Этот проект на основе анализа данных и машинного обучения классифицирует клиентов авиакомпании по их удовлетворенности услугами. Мы используем логистическую регрессию для предсказания, будет ли клиент доволен полетом или нет, на основе различных факторов.

## Установка

Для начала работы с проектом убедитесь, что у вас установлен Python 3.6 или выше. Затем создайте виртуальное окружение и активируйте его:

```bash
python -m venv venv
source venv/bin/activate  # Для Windows используйте venv\Scripts\activate
```

Установите необходимые библиотеки:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn streamlit
```

## Использование

1. Запустите Streamlit приложение:

```bash
streamlit run app.py
```

2. Откройте браузер и перейдите по адресу `http://localhost:8501`, чтобы получить доступ к интерфейсу вашего приложения.

3. Заполните параметры на боковой панели и нажмите на кнопку, чтобы получить предсказание о удовлетворенности клиента.

## Описание кода

### Импорт библиотек

Мы начинаем с импорта необходимых библиотек, включая `numpy`, `pandas`, `seaborn`, `matplotlib` и `scikit-learn`.

### Загрузка данных

Данные о клиентах загружаются из CSV файла:

```python
df = pd.read_csv("https://raw.githubusercontent.com/evgpat/stepik_from_idea_to_mvp/main/datasets/clients.csv")
```

### Предобработка данных

- Удаление ненужных колонок.
- Заполнение пропусков в категориальных признаках значением 'unknown'.
- Замена пропусков в числовых признаках на средние значения.

### Обучение модели

Мы обучаем модель логистической регрессии, чтобы предсказать удовлетворенность клиентов на основе различных факторов.

```python
from sklearn.linear_model import LogisticRegression
auxiliary_model = LogisticRegression()
auxiliary_model.fit(X, y)
```

### Прогнозирование

После обучения модели мы можем использовать её для прогнозирования удовлетворенности клиентов, а также для получения вероятности предсказания.

### Сохранение модели

Модель сохраняется в файл, чтобы её можно было загрузить позже без повторного обучения:

```python
from pickle import dump
with open('data/model_weights.pkl', 'wb') as file:
    dump(main_model, file)
```

## Вклад

Если вы хотите внести свой вклад в проект, пожалуйста, создайте новый репозиторий, внесите ваши изменения и создайте pull request.

## Лицензия

Этот проект лицензируется на условиях MIT License - смотрите файл [LICENSE](LICENSE) для подробностей.
