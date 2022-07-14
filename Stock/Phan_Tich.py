import pandas as pd
import numpy as np

def ReadData(ticker):
    df=pd.read_csv(ticker)
    df.columns=['ticker','date','open','high','low','close','volume']
    df=df.set_index('date').sort_values('date',axis=0,ascending=True)
    df.index=pd.to_datetime(df.index,format="%Y%m%d")
    df['20D-TB']=df['close'].rolling(20).mean()
    df['upper']=df['20D-TB']+2*(df['close'].rolling(20).std())
    df['lower']=df['20D-TB']-2*(df['close'].rolling(20).std())
    df['simple return']=df['close']/df['close'].shift(1)-1
    return df



def ReadDataFromInvestingWeb(ticker):
    df=pd.read_csv(ticker,thousands=',')
    df.columns=['date','close','open','high','low','volume','%change']
    df=df.drop('%change',axis=1)
    df=df.set_index('date')
    df.index=pd.to_datetime(df.index)
    df=df.sort_values('date',axis=0,ascending=True)
    df['20D-TB']=df['close'].rolling(20).mean()
    df['upper']=df['20D-TB']+2*(df['close'].rolling(20).std())
    df['lower']=df['20D-TB']-2*(df['close'].rolling(20).std())
    df['simple return']=df['close']/df['close'].shift(1)-1
    return df


