import pandas as pd

def remove_rows():
    df = pd.read_csv("Texas Weather 07.csv")
    print(df.head())
    df.dropna(axis=0, how='any', subset=['TAVG', 'TMAX', 'TMIN'], inplace=True)
    # df.drop(columns=['STATION', 'NAME', 'DAEV','DAPR','EVAP',	'MDEV',	'MDPR',	'MNPN',	'MXPN',	'PRCP',
    #                       	'SNOW',	'SNWD', 'TOBS'], inplace=True)
    df.drop(columns=['STATION', 'NAME', 'LATITUDE', 'LONGITUDE', 'EVAP', 'ELEVATION', 'MNPN', 'MXPN', 'PRCP', 'TOBS'], inplace=True)

    df.to_csv('TX07Updated.csv', index=False, na_rep='NA')

remove_rows()
