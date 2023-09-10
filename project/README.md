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


# CUTOFF??

trafikk data is only in the range:
*2015-07-16 15:00:00* - *2022-12-31 00:00:00*
meanwhile weather data is in much longer, we gotta remove all where the data is empty

Solution -> merge the frames and drop all rows where 
Trafikkmengde_Totalt_i_retning_Florida is empty, we cant train with that!


## Data loss?

The final *out_file.csv* has 65266 lines of data 
How did we get here?

### trafikkdata.csv
**trafikkdata.csv** has *348641* lines

Looking at **trafikkdata.csv**, we see that there is a lot of missing data 
between 2023-07-01 and 2023-01-01
348641 - 326809 = 21832 lines with no data

so now we only care about 326809 of the lines

but each hour has 5 values, but we only really care bout two of these (florida,danmarksplass)
the 3 other values are the same, or a combination of florida+danmarkplass
so (326809 / 5) * 2 = *130723* lines

After transforming the data and pivoting it so that 
Trafikkmengde_Totalt_i_retning_Danmarksplass and Trafikkmengde_Totalt_i_retning_Florida
are coloums instead of values, with these coloums contaning the data in the coloum "trafikkmengde" for their category, this again splits the amount of lines in two, as now for one hour, we can see both values for florida and danmarkplass!

130723 / 2 = *65361*

This number lines quite nicely up with the amount in our final out file: *65266*

### florida.csv

an average **florida.csv** file has *~52500* lines

but each hour has 6 values (00:00,00:10,00:20.. etc)

In order to align florida with trafikkdata, each hour should have one value, therefore the average (mean) of the value across the hour is taken as the value for that hour, 
this cuts files down to 

52500/6 = ~8750 lines
There are 14 florida files
14*8750 = ~122500 lines altogether

However, florida data files contain weather data from 2010-2023 (halfway through 2023) while traffic data only goes between 2015(07-16)-2022, this means that only
**7.5** (since only half of 2015 file is used) of the florida files are actually used, the rest is cut out by missing values in trafikk.csv. This is done since we want the model to not rely only on weather data, and lots of missing data can really effect modeling. 
There is a point to be made for creating artifical traffic data for previous years, but ironically this is what the model is trying to do anyway (only with future years).

So, 
52500/6 = ~8750 lines
There are **7.5** florida files in use
8*8750 = ~65625 lines altogether

This aligns nicely with our previous estimate of *65266* traffic data lines.

The (65625-65266)=359 line discrepancy arises because of missing data in trafikkdata.csv
For example, between 2015-08-20 01:00:00 and 2015-08-20 13:00:00, all traffic data is missing
There are atleast 111 more cases of this (`using ctrl+f`), leading to 359-111 = 248 line difference


#CHECK_MAIN



big data jump!!!

gap in trafikkdata
2015-08-20T02:00+02:00
to
2015-08-20T12:00+02:00





