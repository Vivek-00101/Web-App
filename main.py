import fbprophet
import streamlit as st
from datetime import date

import requests
from streamlit_lottie import st_lottie

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

st.set_page_config(page_title='Stock Prediction' ,page_icon='chart_with_upwards_trend' ,layout='wide')

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# animation
lottie_coding = load_lottieurl("https://assets10.lottiefiles.com/private_files/lf30_wypj5bum.json")

lottie_coding_2 = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_q7Cm00.json")


START = "2009-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
with st.container():

    left_column, right_column = st.columns(2)

with left_column:
    st.title('Stock Price Prediction')
with right_column:
    st_lottie(lottie_coding_2, height=150, key="stock_2")
st.write("_____")

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME', 'AMZN', 'TSLA')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())
st.write("_____")



# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

with st.container():
    st.write("____")
    left_column, right_column = st.columns(2)
with left_column:
    plot_raw_data()
with right_column:
    st_lottie(lottie_coding, height=300, key="stock")


# Predict forecast with Prophet.


df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)
st.write("_____")
# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("_____")

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
st.write("_____")
