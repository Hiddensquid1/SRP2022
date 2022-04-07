# Importér nødvendige biblioteker
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
plt.rcParams['figure.figsize'] = [20, 5]

# Indlæs data fra CSV-fil (Kilde: Yahoo Finance)
btc_yahoo = pd.read_csv("BTC-USD.csv")

# Eksempel:
btc_yahoo

# Hvis vi kun er interesseret i "Date" og "Adj Close"
btc_yahoo[["Date", "Adj Close"]]

# Datoerne langs x-aksen -- og de justerede lukke-kurser op ad y-aksen
x = btc_yahoo["Date"]
y = btc_yahoo["Adj Close"]

# Plot dataene
plt.xlabel('Dato')
plt.ylabel('Pris (i USD)')
plt.xticks(rotation = 45)
plt.xticks(range(0, int(366), 30))
plt.plot(x, y)

# Forudsig BTC 7 dage ud i "fremtiden"
forudsig_bitcoin = 7
btc_yahoo['Prediction'] = btc_yahoo[['Adj Close']].shift(-forudsig_bitcoin)

# Modellen .fit()'es på lukke-kurserne, hvor der testes på de resterende 15%
X_bitcoin = np.array(btc_yahoo[['Close']])
y_bitcoin = btc_yahoo['Prediction'].values

X_bitcoin = X_bitcoin[:-forudsig_bitcoin]
y_bitcoin = y_bitcoin[:-forudsig_bitcoin]

x_train_btc, x_test_btc, y_train_btc, y_test_btc = train_test_split(X_bitcoin, y_bitcoin,test_size=0.15)
lr_bitcoin = LinearRegression()
lr_bitcoin.fit(x_train_btc, y_train_btc)

# Der giver umiddelbart en 'score' på ~78%
lr_confidence_bitcoin = lr_bitcoin.score(x_test_btc, y_test_btc)
print(lr_confidence_bitcoin*100,'%')

# Resultaterne for modellens forudsigelse
x_projection_bitcoin = np.array(btc_yahoo[['Close']])[-forudsig_bitcoin:]
lr_prediction_bitcoin = lr_bitcoin.predict(x_projection_bitcoin)

for i, k in enumerate(lr_prediction_bitcoin):
    print('Den {} burde prisen (USD) være {}'.format(btc_yahoo["Date"][359+i],k))
