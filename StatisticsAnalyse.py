import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Dataset = pd.read_csv("CarPrice_Assignment.csv")
#Drop as trash
Dataset = Dataset.drop(['car_ID', 'symboling', 'CarName'], axis=1)
#Factorization
ColumnsToFactorize = ['fueltype', 'aspiration', 'doornumber',
                      'carbody', 'drivewheel', 'enginelocation',
                      'enginetype', 'cylindernumber', 'fuelsystem']
Dataset[ColumnsToFactorize] = Dataset[ColumnsToFactorize].apply(lambda x: pd.factorize(x)[0])
#Drop for bad corr
Dataset = Dataset.drop(['fueltype', 'aspiration', 'doornumber',
                        'carbody', 'carheight', 'enginetype',
                        'fuelsystem', 'stroke', 'compressionratio', 'peakrpm'], axis=1)
#Drop for multicorr
Dataset = Dataset.drop(['citympg', 'carlength', 'curbweight',
                        'horsepower', 'highwaympg', 'carwidth',
                        'drivewheel', 'wheelbase', 'cylindernumber', 'boreratio'], axis=1)
Dataset.info()
sns.heatmap(Dataset.corr(),  vmin=-1, vmax=+1, annot=True, cmap='coolwarm')
plt.show()
ax = Dataset['enginesize'].plot.box()
plt.show()
#IQR = Dataset['enginesize'].quantile(0.75) - Dataset['enginesize'].quantile(0.25)
#Dataset = Dataset[Dataset['enginesize']<=Dataset['enginesize'].quantile(0.75)+1.5*IQR]
