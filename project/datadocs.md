### Docs about raw_data

## Trafikkdata

timestamp

figurd out
timestamp has two seperrators, replace aLL | WITH ;
make timestamp unfiorm to make a good df

# Issues

## With parsing

Some florida files have "Relativ luftfuktighet" some dont!
Is this important?? -> yes i think so but this data only exists
in 2022 and 2023, and the rest of the data sets misses ut

## With the data

Weather data has 5 data points for an hour
Traffic data only has 1!

Solution? -> take the mean?
df_weather_resampled = df_weather.resample('H').mean() # or median()