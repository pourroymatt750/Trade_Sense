import streamlit as st
import datetime
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go
import csv
# Import News API
from newsapi import NewsApiClient
import os
from dotenv import load_dotenv

# Config page
st.set_page_config(page_title="Trader Sense", layout="wide")
# Bootstrap CSS
st.markdown(f"""
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css" rel="stylesheet"
integrity="sha384-KK94CHFLLe+nY2dmCWGMq91rCGa5gtU4mk92HdvYe+M/SXH301p5ILy+dN9+nJOZ" crossorigin="anonymous">
""", unsafe_allow_html=True)
# News API key
load_dotenv()
news_api_key = os.getenv('NEWS_API_KEY')
newsapi = NewsApiClient(news_api_key)
# Parse through CSV file to get ticker and company name
stock_tickers = []
stock_names = dict()
with open('sp-500.csv', 'r') as csv_file:
    for line in csv.reader(csv_file):
        stock_tickers.append(line[0])

        stock_names[line[0]] = line[1]
# Calculate beginning of news cycle
tod = datetime.datetime.now()
d = datetime.timedelta(days=28)
a = tod - d
START_NEWS = a.date()
# START is how far data goes back to
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Header
with st.container():
    st.markdown("<h1 style='text-align: center; margin-bottom: 5%; margin-top:-3%;'>Trader Sense</h1>",
                unsafe_allow_html=True)
    left_column, right_column = st.columns(2)
    with left_column:
        selected_stock = st.selectbox('Select a company', stock_tickers)
        st.subheader(stock_names[selected_stock])
    with right_column:
        n_years = st.slider('Years of prediction: ', 1, 3)
        period = n_years * 365
        str_yrs = str(n_years)
        if n_years == 1:
            st.subheader(str_yrs + " year")
        else:
            st.subheader(str_yrs + " years")


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        data_load_state = st.text("Loading data...")
        data = load_data(selected_stock)
        data_load_state.text("")

        st.subheader("Raw data")
        st.write(data.tail())
    with right_column:
        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
            fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
            st.plotly_chart(fig, use_container_width=True)


        plot_raw_data()

# Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns=({"Date": "ds", "Close": "y"}))

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write('forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1, use_container_width=True)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)

# /v2/top-headlines from News API
top_headlines = newsapi.get_top_headlines(q=stock_names[selected_stock],
                                          category='business',
                                          language='en',
                                          )


# Boostrap card to hold the news stories
def news_story(link, image, source, title):
    return f"""
    <div class="card" style="width: 20rem; max-width: 100%; margin-top: 5%; display: flex; justify-content: center;">
        <a style="text-decoration: none;" href="{link}">
          <img id="headline-pic" style="height: 15rem;" src="{image}" class="card-img-top" alt="..." 
          onerror="standby()">
          <div class="card-body">
            <h5 style="color: black; font-size: 135%;" class="card-title">{source}</h5>
            <p style="font-size: 115%;" class="card-text">{title}</p>
          </div>
        </a>
    </div>
    """


st.markdown("<h1 style='text-align: center; margin-top: 5%;'>Recent Headlines</h1>",
            unsafe_allow_html=True)
left = []
mid = []
right = []
with st.container():
    left_column, mid_column, right_column = st.columns(3)
    # Iterate through all news articles
    for headline in top_headlines['articles']:
        # Google News only reposts other news stories, so it ends up being duplicates
        if headline['source']['id'] != 'google-news' and headline['urlToImage'] is not None:
            if headline not in left and len(left) <= len(mid) and len(left) <= len(right):
                with left_column:
                    st.markdown(news_story(
                        headline['url'],
                        headline['urlToImage'],
                        headline['source']['name'],
                        headline['title']
                    ), unsafe_allow_html=True)
                left.append(headline)
            elif headline not in mid and len(mid) <= len(right) and len(mid) <= len(left):
                with mid_column:
                    st.markdown(news_story(
                        headline['url'],
                        headline['urlToImage'],
                        headline['source']['name'],
                        headline['title']
                    ), unsafe_allow_html=True)
                mid.append(headline)
            elif headline not in right and len(right) <= len(left) and len(right) <= len(mid):
                with right_column:
                    st.markdown(news_story(
                        headline['url'],
                        headline['urlToImage'],
                        headline['source']['name'],
                        headline['title']
                    ), unsafe_allow_html=True)
                right.append(headline)

    if not top_headlines['articles']:
        st.markdown("<h3 style='text-align: center; margin-top: 5%;'>No headlines available</h3>",
                    unsafe_allow_html=True)
