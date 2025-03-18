import pandas as pd

def remove_rows():
    df = pd.read_csv("Raw Weather/Oklahoma Weather 08.csv")
    print(df.head())
    df.dropna(axis=0, how='any', subset=['TAVG', 'TMAX', 'TMIN'], inplace=True)
    # df.drop(columns=['STATION', 'NAME', 'DAEV','DAPR','EVAP',	'MDEV',	'MDPR',	'MNPN',	'MXPN',	'PRCP',
    #                       	'SNOW',	'SNWD', 'TOBS'], inplace=True)
    df.drop(columns=['STATION', 'NAME', 'TOBS'], inplace=True)

    df.to_csv('Cleaned Weather/OK08Updated.csv', index=False, na_rep='NA')

remove_rows()
