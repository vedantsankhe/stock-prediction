import streamlit as st
import yfinance as yf
from datetime import date


from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

import numpy as np
import matplotlib.pyplot as plt


st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Stock Forecast App')

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

selected_stock = (st.text_input("Enter Stock Name")).upper()

def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

if selected_stock != "":
  data_load_state = st.text('Loading data...')
  data = load_data(selected_stock)
  data_load_state.text('Loading data... done!')
  st.subheader('Raw data')
  st.write(data.tail())
  st.subheader('Time Series Data')
  plot_raw_data()


def forecast():
  st.write('Forecasting')
  df_train = data[['Date','Close']]
  df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

  m = Prophet()
  m.fit(df_train)
  future = m.make_future_dataframe(periods=365)
  forecast = m.predict(future)

  st.subheader('Forecast data')
  st.write(forecast.tail())

  st.write(f'Forecast plot for 1 years')
  fig1 = plot_plotly(m, forecast)
  st.plotly_chart(fig1)

def sma(data, peroid=30,Column="Close"):
    return data[Column].rolling(window=peroid).mean()

  
def predict():
  st.write('Prediction')
  stock_df = data
  stock_df["SMA20"] = sma(stock_df,20)
  stock_df["SMA50"] = sma(stock_df,50)

  stock_df["Signal"] = np.where(stock_df['SMA20'] > stock_df['SMA50'],1,0)
  stock_df["Position"] = stock_df["Signal"].diff()

  stock_df["Buy"] = np.where(stock_df['Position'] == 1, stock_df["Close"], np.NAN)
  stock_df["Sell"] = np.where(stock_df['Position'] == -1, stock_df["Close"], np.NAN)

  plt.title('Volume of Stock traded')
  plt.ylabel("Closing Price")
  plt.xlabel("Date")
  plt.plot(stock_df['Close'],alpha= 0.5,  label = "close")
  plt.plot(stock_df['SMA20'],alpha= 0.5,  label = "SMA20",color="green")
  plt.plot(stock_df['SMA50'],alpha= 0.5,  label = "SMA50")
  plt.scatter(stock_df.index,stock_df["Buy"],alpha=1,label = "Buy Signal",marker = "^", color="green")
  plt.scatter(stock_df.index,stock_df["Sell"],alpha=1,label = "Sell Signal",marker = "v", color="red")
  st.pyplot()

  

def get_macd(DF, a, b, c):
    df = DF.copy()
    df['MA Fast'] = df['Adj Close'].ewm(span=a, min_periods=a).mean()
    df['MA Slow'] = df['Adj Close'].ewm(span=b, min_periods=b).mean()
    df["MACD"] = df['MA Fast'] - df['MA Slow']
    df['Signal'] = df.MACD.ewm(span=c, min_periods=c).mean()
    df["Histrogram"] = df.MACD - df.Signal
    return df

#plotting macd graph
def plot_macd(prices, macd, signal, hist):
    ax1 = plt.subplot2grid((8,1), (0,0), rowspan = 5, colspan = 1)
    ax2 = plt.subplot2grid((8,1), (5,0), rowspan = 3, colspan = 1)

    ax1.plot(prices)
    ax2.plot(macd, color = 'grey', linewidth = 1.5, label = 'MACD')
    ax2.plot(signal, color = 'skyblue', linewidth = 1.5, label = 'SIGNAL')

    for i in range(len(prices)):
        if str(hist[i])[0] == '-':
            ax2.bar(prices.index[i], hist[i], color = '#ef5350')
        else:
            ax2.bar(prices.index[i], hist[i], color = '#26a69a')

    plt.legend(loc = 'lower right')
    st.pyplot()

#trading strategy
def implement_macd_strategy(prices, data):
    buy_price = []
    sell_price = []
    macd_signal = []
    signal = 0

    for i in range(len(data)    ):
        if data['MACD'][i] > data['Signal'][i]:
            if signal != 1:
                buy_price.append(prices[i])
                sell_price.append(np.nan)
                signal = 1
                macd_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                macd_signal.append(0)
        elif data['MACD'][i] < data['Signal'][i]:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(prices[i])
                signal = -1
                macd_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                macd_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            macd_signal.append(0)

    return buy_price, sell_price, macd_signal
    


def predict_macd():
   st.write('Buy and Sell Call Using MACD')
   plt.rcParams['figure.figsize'] = (20, 10)
   plt.style.use('fivethirtyeight')
   macd_df = get_macd(data, 26, 12, 9)
   plot_macd(data['Close'], macd_df['MACD'], macd_df['Signal'], macd_df ['Histrogram'])
   buy_price, sell_price, macd_signal = implement_macd_strategy(data['Close'], macd_df)

   ax1 = plt.subplot2grid((8, 1), (0, 0), rowspan=5, colspan=1)
   ax2 = plt.subplot2grid((8, 1), (5, 0), rowspan=3, colspan=1)
   ax1.plot(data['Close'], color='skyblue', linewidth=2, label='Stock')
   ax1.plot(data.index, buy_price, marker='^', color='green', markersize=10, label='BUY SIGNAL', linewidth=0)
   ax1.plot(data.index, sell_price, marker='v', color='r', markersize=10, label='SELL SIGNAL', linewidth=0)
   ax1.legend()
   ax1.set_title(' MACD SIGNALS')
   ax2.plot(macd_df['MACD'], color='grey', linewidth=1.5, label='MACD')
   ax2.plot(macd_df['Signal'], color='skyblue', linewidth=1.5, label='SIGNAL')

   for i in range(len(macd_df)):
    if str(macd_df['Histrogram'][i])[0] == '-':
        ax2.bar(macd_df.index[i], macd_df['Histrogram'][i], color='#ef5350')
    else:
        ax2.bar(macd_df.index[i], macd_df['Histrogram'][i], color='#26a69a')

   plt.legend(loc='lower right')
   st.pyplot()


option = st.sidebar.selectbox("",("Stock Data","Forecast","Buy And Sell Calls","MACD"))

if "Forecast" in option:
  forecast()
if "Buy And Sell Calls" in option:
  predict()
if "MACD" in option:
  predict_macd()


