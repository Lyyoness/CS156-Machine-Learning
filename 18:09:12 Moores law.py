import pandas as pd
import numpy as np
from scipy import stats
import datetime as dt 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


benchmarks = pd.read_csv('benchmarks.csv')
print benchmarks.head(10)


#extracting the date
date = benchmarks['testID']
date = date.str.extract(r'(-[0-9]+-)', expand=False)
benchmarks['testID'] = date.str.replace('-', '')
benchmarks = benchmarks.dropna()
benchmarks['testID'] = benchmarks['testID'].apply(lambda x: pd.to_datetime(x).strftime('%m/%d/%Y'))

# extracting tomcat performances & sorting by dates
tomcatv = benchmarks[benchmarks['benchName'] == '101.tomcatv']
tomcatv = tomcatv.sort_values(['testID'])
print tomcatv.head()

#plotting the general dates
x = tomcatv['testID']
y = tomcatv['base']
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.gcf().autofmt_xdate()
# plt.yscale('log')

#training and plotting model
tomcatv['testID'] = pd.to_datetime(tomcatv['testID'])
tomcatv['testID'] = tomcatv['testID'].map(dt.datetime.toordinal)
x2 = tomcatv['testID']
slope, intercept, _, _, _ = stats.linregress(x2, y)
line = slope*x2+intercept
plt.plot(x, y, 'b', x, line, '-k')
plt.show()

