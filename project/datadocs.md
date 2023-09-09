### Docs about raw_data

## Trafikkdata

timestamp

figurd out
timestamp has two seperrators, replace aLL | WITH ;
make timestamp unfiorm to make a good df

# Issues

## With parsing

"< 5,6m",    IS SUPER IMPORTANT

Some florida files have "Relativ luftfuktighet" some dont!
Is this important?? -> yes i think so but this data only exists
in 2022 and 2023, and the rest of the data sets misses ut

## With the data

Weather data has 5 data points for an hour
Traffic data only has 1!

Solution? -> take the mean?
df_weather_resampled = df_weather.resample('H').mean() # or median()


# CUTOFF??

trafikk data is only in the range:
*2015-07-16 15:00:00* - *2023-01-01 00:00:00*
meanwhile weather data is in much longer, we gotta remove all where the data is empty

Solution -> drop all rows where 
Trafikkmengde_Totalt_i_retning_Florida is empty, we cant train with that!



**TODO**: change date to better to train  