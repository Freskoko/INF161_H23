### TODO

INCLUDE THE GRAPHS IN DISUCSSION TO SHOW RUSH HOURS !!!

CHANGE THE PART ABOUT DATA FLCUTATION TO A BETTER DATE AND TIME !!!



### REPORT

- update data loss parsing section

- update report for desci0ns made to be reflected in data not just because sounds good

- evulate multiple models

- RUN AGAIN MANY TIMES FIND GOOD
    - RUN WITH DIFFERENT KNN
    - RUN WITH NORMALIZATION, AND NOT


- NETTSIDE
    - DE GIR ALT AV DATA, OG SÅNN 


- graph weekly traffic amounts
- grap monthly traffic amounts


### CODE

- mkdirs

- cleanup files that may not be used

- cleanup

- conda and lock file

- put into one giant python file

# Report for INF161 cycle traffic project

***If I'm not in PDF format, please open me in markdown formatting! (in vscode there is a button in the top right.)***

*Henrik Brøgger*

This report pertains to the INF161 project this fall, in creating a model which can guess bike traffic across nygårdsbroen.

### Index:

- Data Exploration
- Removed data / Data loss
- Feature engineering
- Issues/Choices
- Data exploration post processing
- Model choice and evaluation
- Results
- Conclusion



# Data exploration:
In src/figs, there are images presenting each of the coloums in the final data frame, plotted against the total amount of traffic. This part of the report explores these figures.

Data is observed using spearmann and pearson correlation.


-------------

### Globalstråling

**Raw-observation**

![globalstråling vs traffik](src/figs_new/GlobalstralingVSTotal_trafikk_PRE_CHANGES.png)

Looking at the figure above , it is clear that the data contains outliers, as a the amount of global radation cannot exceed many thousands. #TODO REFRENCE

**Post-processing**

![globalstråling vs traffik](src/figs_new/GlobalstralingVSTotal_trafikk_POST_CHANGES.png)

After processing this data, treating outliers, one can see that the data sits between values of .-4 to 900.

It is clear that there is some correlation between globalstråling and cycle-traffic. 

There seems to be little difference in correlation when globalstråling lies between 0-400, but in values over this, and especially over 600, traffic decreases. 

With a pearson corr value of *0.2985*, this is decently strong, but can still be a good indicator of correlation.

The spearmann correlation value of *0.4716* is a good sign. Due to the nature of the data "jumping"  up and down, (meaning that that for one given globalstråling `n`, it can have values a 300, while `n+1` has 150, and `n+2` has 300 again). The spearmann value may not be as useful here, as it is most useful when observing monotonic data.

-------------

### Lufttemperatur

**Raw-observation**

![lufttemp vs traffik](src/figs_new/LufttemperaturVSTotal_trafikk_PRE_CHANGES.png)

Looking at the figure above , it is clear that the data contains outliers, and the data seems to pool weirdly around certain values. This needs to be adjusted for. 

**Post-processing**

![lufttemp vs traffik](src/figs_new/LufttemperaturVSTotal_trafikk_POST_CHANGES.png)

After processing this data, treating outliers, one can see that the data sits between values of -10-32.

It is clear that there is some correlation between temperature and cycle-traffic. 

There seems to be little difference in correlation when lufttemperatur lies between 3-20, but in values over 20, and values under 3, traffic decreases. 

With a pearson corr value of *0.2783*, this decently strong, but can still be a good indicator of correlation.

The spearmann correlation value of *0.3405* is a good sign. However, this data is not monotic, as it goes up, then down later. This means that the spearmann correlation cannot be trusted to a large extent.

-------------

### Lufttrykk

**Raw-observation**

![lufttrykk vs traffik](src/figs_new/LufttrykkVSTotal_trafikk_PRE_CHANGES.png)

Looking at the figure above , it is clear that the data contains outliers, and the data seems to pool weirdly around certain values. This needs to be adjusted for. 

**Post-processing**

![lufttrykk vs traffik](src/figs_new/LufttrykkVSTotal_trafikk_POST_CHANGES.png)

Looking at the *Lufttrykk vs Total_trafikk* graph above, it may seem like there is a correlation between the two. It seems that values around 980-1020 provide around the same amount of cyclists. Values higher than 1020 and lower than 980 causes a drop off inn traffic.

With a pearson corr value of *0.0714*, this is not very strong at all, despite what it may seem at first look.
The spearmann correlation value of *0.0818* is also a bad sign, but the spearmann corr may not be as good of an indicator as the pearson corr since this data is not monotonic.

-------------

### Solskinnstid

**Raw-observation**

![solskinn vs traffik](src/figs_new/SolskinstidVSTotal_trafikk_PRE_CHANGES.png)

Looking at the figure above , it is clear that the data contains outliers, and the data seems to pool weirdly around certain values. This needs to be adjusted for. 

**Post-processing**

![solskinn vs traffik](src/figs_new/SolskinstidVSTotal_trafikk_POST_CHANGES.png)

Looking at the *Solskinn vs Total_trafikk* graph above, it may be hard to spot a correlation between the two. It seems that solskinnstid does not effect the amount of cyclists. 

However, with a pearson corr value of *0.2623*, this indicates atleast a casual correlation. This data will still be useful.
The spearmann correlation value of *0.3616*  is ok, but the spearmann corr may not be as good of an indicator as the pearson corr since this data is not monotonic.

-------------

### Vindretning

**Raw-observation**

![vindretning vs traffik](src/figs_new/VindretningVSTotal_trafikk_PRE_CHANGES.png)

Looking at the figure above , it is clear that the data contains outliers, and the data seems to pool weirdly around certain values. This needs to be adjusted for. 

**Post-processing**

![Vindretning vs traffik](src/figs/VindretningVSTotal_trafikk.png)

Looking at the *Vindretning vs Total_trafikk*  graph above, it is up for argument if there is a strong correlation between the two, but there is some data that can be useful.
It seems that between x=100-350 values are pretty much consistent, however a drop is seen at around 250. Values between 100-0 are also very very low, and could be reflective of something else? Vindretning and traffic seem to be correlated, but these vindretning values can be further processed, to try to extract further data from the wind.  

![vindretning vs traffik](src/figs_new/Vindretning_xVSTotal_trafikk_POST_CHANGES.png)

![vindretning vs traffik](src/figs_new/Vindretning_yVSTotal_trafikk_POST_CHANGES.png)

This data has been transformed quite a bit. Vindretning was originally a number between 0-360, and has transformed to two values. The degrees (0-360) can be imagined as points on a unit circle. 
Converting this point to two sperate values, x and y reveal more about the nature of the wind. Originally only the wind direction was known, but now the wind x and y directions are known, or atleast simulated.

Mathematically speaking:
```
    df["Vindretning_radians"] = np.radians(df["Vindretning"])
    df["Vindretning_x"] = np.cos(df["Vindretning_radians"])
    df["Vindretning_y"] = np.sin(df["Vindretning_radians"])

```

Looking at the *Vindkast x/y vs Total_trafikk* graphs above, it is not right away clear that vindkast has a correlation with cycle traffic.
The original undivided data had a slightly positive pearson correlation of *0.139*, but now after splitting the data in two, it is clear that *Vindretning_x* has a positive pearson corr of *0.1283* while *Vindretning_y* has a negative correlation of *-0.1107*. Splitting this value into two allowed us to gain a deeper understanding of this value, understanding that some vindretning is negativley correlated!
Since vindretning has been transformed to two different variables, the original "Vindretning" has been dropped.

-------------

### Vindkast 

**Raw-observation**

![vindkast vs traffik](src/figs_new/VindkastVSTotal_trafikk_PRE_CHANGES.png)

Looking at the figure above , it is clear that the data contains outliers, and the data seems to pool weirdly around certain values. This needs to be adjusted for. 

**Post-processing**

![vindkast vs traffik](src/figs_new/VindkastVSTotal_trafikk_POST_CHANGES.png)

Looking at the *Vindkast vs Total_trafikk* graph above, it is clear that vindkast has a correlation with cycle traffic.
Values bteween 0-15 dont seem to effect traffic, but values above 15 m/s indicate strong winds and therefore we see a drop in traffic at these values.

However, with a pearson corr value of *0.0347*, this indicates quite a weak corrrelation. This data will still be useful.
The spearmann correlation value of *0.109*  is ok, and since this data portrays as somewhat monotonic, could tell us some correlation is present.

-------------

### Vindstyrke

**Raw-observation**

![vindstyrke vs traffik](src/figs_new/VindstyrkeVSTotal_trafikk_PRE_CHANGES.png)

Looking at the figure above , it is clear that the data contains outliers, and the data seems to pool weirdly around certain values. This needs to be adjusted for. 

**Post-processing**

![vindstyrke vs traffik](src/figs_new/VindstyrkeVSTotal_trafikk_POST_CHANGES.png)


![vindstyrke vs vindkast](src/figs_new/VindstyrkeVSVindkast_POST_CHANGES.png)


The *Vindkast* and *Vindstyrke* variables have a pearson correlation of 0.979, for the purpouses of the data, they tell us virtually the same thing. 

*Vindstyrke* has a correlation of 0.0321 with *Total trafikk*, 
while *Vindkast* has a correlation 0.0325 with *Total trafikk*.
*Vindstyrke* has a pearson correlation which is 0.004 less than *Vindstyrke*, this is almost nothing, but for the purpouses of this paper, i choose to keep *Vindkast*. It is also important to note that when two variables are so similar and correlate so well, it is like giving the model the same data two times, which can lead to unwanted effects, like the model over-weighting these two factors or otherwise underrelying on one, and overrelying on the other. When they tell the same story, there is no need to keep them both.


-------------

# Variations within time

![year](src/figs_new/monthly_traffic.png)

Certain months have different amounts of mean traffic, so providing the model the month will help it understand this correlation. I am using dummies from python in order to setup a coloumn for each month. 

![week1](src/figs_new/weekly_traffic.png)

Certain days have different amounts of mean traffic, so providing the model the day will help it understand this correlation. I am using dummies from python in order to setup a coloumn for each day. 

### Yearly variations of traffic data / Correlation of the two directions

![FloridaDanmarksplass vs time](src/figs/timeVStrafficBoth.png)

Looking at the *FloridaDanmarksplass vs time* graph above, one can see that the two variables describing the amount of people driving each direction are very correlated.

Both of the statistical tests back this up aswell, having high values of: pearson = *0.0407* and spearmann = *0.8269
It is for this reason i have chosen to combine the two variables into one, as a "total traffic variable". 

This graph also visualizes a large cycling peak in 2017, due to a large bicycle competion happening that year. 
This is the cause of a great deal of outliers. The solution to this is removing data which sits in the 99th percentile. The model does not need to be good at guessing when the next large scale bicycling competiton is, it is more about day to day cycling. 


-------------

### Correlation matrix

**Raw-observation**

![Corr matrix](src/figs_new/corr_matrix_PRE_CHANGES.png)

Looking at the *Corr matrix* graph above, it tells us that the data needs to be processed, as values are all over the place, most probably due to outliers.

**Post-processing**

![Corr matrix](src/figs_new/corr_matrix_POST_CHANGES.png)

The variables *Globalstråling* and *Solskinnstid* have a high degree of correlation, at 0.68.
This is high, but not high enough that they tell us the same thing, so i am going to keep both variables. It is also important to note that both variables have a decent degree of correlation with *Total_trafikk*, so they could both be very important.

The variables *Luftemperatur* and *Globalstråling* are also quite correlated, as expected, but they only have a pearson correlation of *0.41*, so keeping both values here, (espeically since they both correlate so well with *Total_trafikk*) is the correct choice. 

The variables which seem to have a good correlation with *Total_trafikk* are:

- Globalstråling (**0.29**)

- Solskinnstid (**0.24**)

- Lufttemperatur (**0.26**)

- Vindretning_x/Vindretning_y (~ +/- **0.12~**)


### Dropped coloumns

- *Vindstyrke*

<p>
Vindstyrke and vindkast have a high degree of correlation, and statistically, it is like having the same variable two times. The two variables also have a very similar correlation with traffic amounts. Vindstyrke was dropped, as vindkast had a slightly higher correlation to traffic. 
<p>

- *Vindretning/Vindretning_radians*

<p>
These have been transformed to Vindretning_x and Vindretning_y which provide more information about the variables.
<p>

- *Relativ luftfuktighet*

<p>
Drop "Relativ luftfuktighet" as this data only exists in 2022 and 2023. While this would be very valuable, its hard to train a dataset with a lot of missing data. 
<p>

- *Data in traffic files*

``` json

    columns=[
        "Trafikkregistreringspunkt",
        "Navn",
        "Vegreferanse",
        "Fra",
        "Til",
        "Dato",
        "Fra tidspunkt",
        "Til tidspunkt",
        "Dekningsgrad (%)",
        "Antall timer total",
        "Antall timer inkludert",
        "Antall timer ugyldig",
        "Ikke gyldig lengde",
        "Lengdekvalitetsgrad (%)",
        "< 5,6m",
        ">= 5,6m",
        "5,6m - 7,6m",
        "7,6m - 12,5m",
        "12,5m - 16,0m",
        ">= 16,0m",
        "16,0m - 24,0m",
        ">= 24,0m",
    ]

```

These coloumns do not really tell us much, and could really just confuse the model. 


### Dropped values - updates vlaues 
### specify if its test training or validation 

Values that were deemed as outliers or "99999" were transformed into appropriate values by a KNNImputer, with settings *weights* = distance, since this pertains to date data. When mentioning that "x" data points were transformed to NaN, this includes the 99999 data points as well as those outside the borders specified in each section.

```
Number of NaNs in each column: (training data)
Globalstraling              85
Solskinstid                 94
Lufttemperatur             169
Vindretning                278
Vindstyrke                 169
Lufttrykk                  169
Vindkast                   169
Relativ luftfuktighet    45746
```


- *Globalstråling*

<p> Values over 1000 in **Globalstraling** are considered malformed. </p>
114 data points are transformed into NaN.

This value was chosen because values over this are only observed "ved atmosfærenses yttergrense"

[ref]("https://veret.gfi.uib.no/") 

- *Solskinnstid*

<p>
Values above 10.01 in **Solskinstid** are are considered  malformed</p>
123 data points are transformed into NaN. 

The solskinstid scale is between 0-10

[ref]("https://veret.gfi.uib.no/") 

- *Lufttrykk*

<p>
Values above 1050 in **Lufttrykk** are considered malformed. </p>
506 data points turn into NaN.

935 and 1050 are the min/max records of all time. 

[ref]("https://en.wikipedia.org/wiki/List_of_atmospheric_pressure_records_in_Europe") 


- *Luftemperatur*

<p>
Values above 37 in **Lufttemperatur** are considered malformed. </p>
507 values are turned into NaN.

Over 37 degrees is not realistic for norway, as the warmest ever recorded was 35.6 degrees

[ref]("https://no.wikipedia.org/wiki/Norske_v%C3%A6rrekorder") 

- *Vindkast*

<p>
Values above 65 in **Vindkast** are considered malformed </p>
506 values are considered NaN.

Over 65m/s is not realistic for norway, as the highest value ever recorded was 64,7m/s

[ref]("https://no.wikipedia.org/wiki/Norske_v%C3%A6rrekorder") 

- *Vindretning*

<p>
Values above 360 in **Vindretning** are considered malformed </p>
669 values are lost.

Since vindretning is measured from 0-360, there is no way a degrees of more than 360 could be measured.

- *Vindstyrke*

Values above 1000 are considered malformed.

There are 506 missing values here, but this col is dropped in favour of "Vindkast" anyway, as they are so correlated.

- *Relativ luftfuktighet*

This coloumn is dropped from the start, since there is so much missing data (56513 NaN values)


# Feature engineering

### These features were added:

- *Hour*
<p> From the date, the hour was added as a coloumn. This can help the model make a link between hour and traffic 
</p>
Range: 0-24

-----------------------------------

- *Day_in_week*
<p> From the date, the day in the week was added, This will help the model make a link between days and traffic
</p>
Range: 0-7

-----------------------------------

- *Month*
<p> From the date, the month was added as a coloumn. This can help the model make a link between time of year and traffic
</p>
Range: 1-12

-----------------------------------

- *Weekend*
<p> From the date, a 0/1 coloumn for if it is a weekend or not was added. This can help the model make a link between time of week and traffic
</p>
Range: 0/1

-----------------------------------

- *Public_holiday*
<p> From the date, a 0/1 coloumn for if it is a public holiday or not was added. This can help the model make a link between specials days of the year and traffic.
</p>
Range: 0/1

-----------------------------------

- *Raining*
<p> From the air pressure, a 0/1 coloumn for if it is raining or not was added. Rain and air pressure are not directly linked, but it may be possible to guess weather from air pressure. Reference:

[Rain link]("https://geo.libretexts.org/Bookshelves/Oceanography/Oceanography_101_(Miracosta)/08%3A_Atmospheric_Circulation/8.08%3A_How_Does_Air_Pressure_Relate_to_Weather)

</p>
Range: 0/1

-----------------------------------

- *Summer*
<p> From the months, a 0/1 coloumn that specificed if it is summer or not was added (June-July)

</p>
Range: 0/1

-----------------------------------

- *Winter*
<p> From the months, a 0/1 coloumn that specificed if it is summer or not was added (October-Feburary)

</p>
Range: 0/1

-----------------------------------

- *Rush hour*
<p> From the months, a 0/1 coloumn that specificed if the hour is a rush hour (7-9 and 15-17)

</p>
Range: 0/1

-----------------------------------

- *Nightime*
<p> From the months, a 0/1 coloumn that specificed if the hour is in the middle of the night (22-6)

</p>
Range: 0/1

-----------------------------------

- *Vindretning_x/Vindretning_y*
<p> Vindretning contains values between 0-360, and these are transformed to points on a circle

```python

    df["Vindretning_radians"] = np.radians(df["Vindretning"])
    df["Vindretning_x"] = np.cos(df["Vindretning_radians"])
    df["Vindretning_y"] = np.sin(df["Vindretning_radians"])

```

</p>
Range: -1/1

-----------------------------------

- *Total_trafikk*
<p> The numbers for the two rows of traffic were combined to one. 

</p>
Range: N/A

-----------------------------------

### Considered Features that were decided against 

- *Total traffic in retning danmarkplass*,
- *Total traffic in retning florida*,

<p> The reason adding this coloumn doesnt work is, well, if we know how much traffic there is, there is no point in guessing how much traffic there is.
</p>

Range: N/A

-----------------------------------

- *Last_Total traffic in retning florida*,
- *Last_Total traffic in retning danmarksplass*,
- *Last_Total traffic*,

<p> This coloumn would be the value for traffic in the previous row.
The reason adding this coloumn doesnt work is that it is much harder to train the model when you have to train one line at a time, and use the last row's value's as training values. 
This could also be a big problem because if we guess wrong on the last traffic, that value will be brought with to the next row's guess, and further for ALL the rows, and if that value is wrong, well then ALL the guesses are potentially wrong. 
</p>

Range: N/A

-----------------------------------
- *Day in month*

<p> This coloumn would tell us what day in the month it is, but this is a bit overkill considering the other values we have, and i dont expect traffic to fluctuate a lot between the start and the end of the month.
</p>

Range : 1-31

### Considered features that were dropped: Normalizing values


- *Globalstraling*

<p>
These values are between 0-1000, and could therefore be normalized to a 0-1 scale without much data loss

However, not much was acheived by doing this.
<p>

- *Luftrykk*

<p>
These values are between 940-1050, and normalizing these values to a 0-1 scale could help the model understand the numbers.
<p>

However, not much was acheived by doing this.

- *Solskinnstid*

<p>
These values are between 0-10, and could therefore be normalized to a 0-1 scale without much data loss
<p>

However, not much was acheived by doing this.

- *Vindkast*

["graph"]("src/figs_new/VindkastVSTotal_trafikk_POST_CHANGES.png")

<p>
These values are between 0-25, but there is a clear link between high vindkast and low traffic, so by squaring the values of vindkast could help the model understand that higher numbers mean a large descrease in traffic, while lower numbers do not have an effect on traffic.

However, not much was acheived by doing this. 

# Issues

## With parsing

- Issue:
trafikkdata.csv has both "|" and ";" as seperators, and cant be directly opened with pd.open_csv().

- Solution:
open the file as a long string, replace all "|" with ";"
save the string as a bytes object using StringIO
open the bytes object with pd.open_csv()

----------------

## With aligning the two files

- Issue:
2022 and 2023 florida weather files have a coloumn called "Relativ luftfuktighet" 
This seems important, and could help the model

- Solution:
Since this is missing for all the other years, it is best just to drop it

----------------

- Issue:
Weather data has 6 data points for an hour
Traffic data only has 1 data point for an hour

- Solution: 
Take the mean of the 6 values in the weather data
example:
`df_weather_resampled = df_weather.resample('H').mean() # or median()`

----------------

- Issue:
Traffic data is only in the range:
*2015-07-16 15:00:00* - *2022-12-31 00:00:00*
meanwhile weather data is in much longer from 2010-2023

- Solution: 
After merging the two frames drop all rows where values in traffic data is empty, such as the col:
`Trafikkmengde_Totalt_i_retning_Florida`

----------------

- Issue:
In *trafikkdata.csv* there are gaps in the data for trafikkmende, such as between
2015-08-20T02:00+02:00
and
2015-08-20T12:00+02:00

- Solution:
These gaps are not big enough to warrant large changes, and data is still found for 08-20 for other years, 
the model should be able to predict fine with some missing data like this. 

----------------

# Data loss

:warning: **THIS NEEDS TO BE UPDATED SINCE PARSING IS NOW A BIT DIFFERENT**



**Below is a walkthrough of how the files have been treated, and where data loss comes would come from**

The final *out_file.csv* has 64652 lines of data (before splitting into train,validation and test data)

This is much less than the **trafikkdata.csv** and **florida.csv** files contain!
How did we get here?

## trafikkdata.csv
**trafikkdata.csv** has *348641* lines

Looking at **trafikkdata.csv**, we see that there is a lot of missing data 
between *2023-07-01* and *2023-01-01*
348641 - 326809 = 21832 lines with no data

We drop these, and we are left with 326809 lines

Each hour has 5 values, but we only really care bout two of these (florida,danmarksplass)
the 3 other values are the same, or a combination of florida+danmarkplass
so (326809 / 5) * 2 = *130723* lines

After transforming the data and pivoting it so that 
```Trafikkmengde_Totalt_i_retning_Danmarksplass``` and ```Trafikkmengde_Totalt_i_retning_Florida```
are coloumns instead of values within the ```felt``` coloumn. 

These two new coloums contaning the data previously in the coloum ```trafikkmengde``` for their category, this again splits the amount of lines in two, as now for one hour, we can see both values for florida and danmarkplass! 
This trick is possible because all other coloums except for 
```trafikkmengde```,
```Trafikkmengde_Totalt_i_retning_Danmarksplass``` and ```Trafikkmengde_Totalt_i_retning_Florida```
were dropped, and the date is the index

130723 / 2 = *65361*

This number lines up almost quite nicely up with the amount in our final out file: *64652*

## florida.csv

an average **florida.csv** file has *~52500* lines

but each hour has 6 values (00:00,00:10,00:20.. etc)

In order to align florida with trafikkdata, each hour should have one value, therefore the average (```mean```) of the value across the hour is taken as the value for that hour, 
this cuts files down to 

52500/6 = ~8750 lines
There are 14 florida files
14*8750 = ~122500 lines altogether

However, florida data files contain weather data from *2010-2023 (halfway through 2023)* while traffic data only goes between *2015(07-16)-2022*, this means that only
**7.5** (0.5 since only about half of the 2015 weather file is used) of the florida files are actually used, and the rest are cut out by missing values for those dates in trafikk.csv. 

- This is done since we want the model to not rely only on weather data, and lots of missing data can really effect modeling. 
- There is a point to be made for creating artifical traffic data for previous years, but ironically this is what the model is trying to do anyway (only with future years).


52500/6 = ~8750 lines
There are **7.5** florida files in use

(7*8750) + (24336/6) = ~65266 lines altogether
**NOTE** <p> (24336/6) is the amount of relevant lines in the 2015 florida weather file
(07-16 to 12-31) <p>

This aligns almost quite nicely with our previous estimate of *64652* traffic data lines.

The discrepency in length comes from severe outliers.
*65266 - 64652 = 614 lines too many*

When dropping unreasonable values in ```dataframe_handling.py```, see above sections for this. 



# RESULTS :

**TODO**
 - Er resultater av tre fundamentalt forskjellige modeller rapportert?
 - Update graphs to be easier to read



![attempt1_MSE](src/figs/MANYMODELS_MSE.png)

One can see that RandomForestRegressor with n_estimators; 200 is the best model with a RMSE of 25.301.

#TODO COMMENT ON MULTIPLE MODELS

---------------------------

After finding the best model, hyper-parameters were found.

![hyperparam](src/figs/MSE_hyperparam_models_V3.png)

Seeminlgy, a higher n_estimators yeiled slightly better results. 

This may need to be explored further, but attempts to train past 250 n_estimators results in a slowed console and after a while the text "Killed", so i think i am pushing my memory too far. Also, it seems turning up the n_esimators more may just lead to diminishing returns.


--------------------------

After finding the best model and seeing how it performed on validation data, we can use the ```best_model.feature_importances_``` ouput to evaluate importance of coloumns.

Model for test data = False
MSE: 524.0117169640222
RMSE: 22.891302212063476
```json
           Feature  Importance
19       rush_hour    0.320061
14         weekend    0.189385
5             hour    0.111220
2   Lufttemperatur    0.104526
20       sleeptime    0.067550
13           month    0.053060
0   Globalstraling    0.035950
3        Lufttrykk    0.024350
4         Vindkast    0.020831
21   Vindretning_x    0.017340
22   Vindretning_y    0.015379
1      Solskinstid    0.012289
15  public_holiday    0.009418
6         d_Friday    0.006057
17          summer    0.003716
7         d_Monday    0.001991
10      d_Thursday    0.001882
11       d_Tuesday    0.001596
12     d_Wednesday    0.001392
18          winter    0.000873
16         raining    0.000638
8       d_Saturday    0.000253
9         d_Sunday    0.000243
```

which is pretty good considerding the ```DummyRegressor``` has a RMSE of ```RMSE: 57.63``` on validation data! 

So, now that we have a model, we can try to tweak it, in order to get better results.
Of course, validation data will be used to see if the model is good or not. Test data is saved entirely for last. 

*Changes to attempt* 
- Adding dummy variables for months
- Data normaliazation
- Changing the n_neigbours for the KNNimputer
- Removing dummy variables for days
- Removing the raining coloumn


### Adding dummy variables for months:

After changing the month coloumn from being a number 0-11, to instead each month haivng their own coloumn with value of either 0 or 1.


Model for test data = False
MSE: 537.0067702399648
RMSE: 23.17340653076204
```json
           Feature  Importance
29       rush_hour    0.320061
24         weekend    0.189387
5             hour    0.113872
2   Lufttemperatur    0.107780
30       sleeptime    0.067550
0   Globalstraling    0.037603
3        Lufttrykk    0.023860
14        m_August    0.022208
4         Vindkast    0.021975
31   Vindretning_x    0.017055
32   Vindretning_y    0.015136
1      Solskinstid    0.012720
25  public_holiday    0.010564
21      m_November    0.005824
6         d_Friday    0.005786
18          m_June    0.003761
20           m_May    0.003385
15      m_December    0.003368
22       m_October    0.002600
10      d_Thursday    0.001983
7         d_Monday    0.001979
11       d_Tuesday    0.001585
28          winter    0.001547
12     d_Wednesday    0.001375
23     m_September    0.001354
13         m_April    0.001340
16      m_Feburary    0.001114
17          m_July    0.000775
19         m_March    0.000734
26         raining    0.000623
27          summer    0.000621
8       d_Saturday    0.000241
9         d_Sunday    0.000238

```

After this change, the RMSE increased by about 0.3, proving that adding dummy variables for the months did not decrease the RMSE.
It is also interesting to note that the same 5 variables stay the most important, but the month variables end up having vastly different importances.

*Important variables*
``` 
29       rush_hour    0.320061
24         weekend    0.189387
5             hour    0.113872
2   Lufttemperatur    0.107780
30       sleeptime    0.067550
```

 August is very important, while March is very unimportant.
When adding dummy variables for months, the summer variable becomes very unimportant, meaning that the model may lean more on the months rather than summer.
This feature may have worked in theory, as adding dummy variables for days does, however the month variables are better reflected in their own coloumn, and through other variables such as summer/winter

### Data normalization

These variables were changed to a 0-1 scale

"Globalstraling",
"Lufttrykk",
"Solskinstid",

The thought behind this is that since these values are all between 0-10 or in the case of Lufttrykk, 950-1050, changing to a 0-1 scale would help the model understand the difference between a high and low value. 

And 

All values of "Vindkast" were taken to the second power.

![graph](src/figs_new/VindkastVSTotal_trafikk_POST_CHANGES.png)

The thought behind this is that since values between 0-15 do not effect traffic, but values between 15-25 do, it would be a way to make the model understand this.  

Results:

Model for test data = False

MSE: 549.9913200328366

RMSE: 23.45189374086529

```
           Feature  Importance
19       rush_hour    0.320061
14         weekend    0.189385
5             hour    0.111220
2   Lufttemperatur    0.104526
20       sleeptime    0.067550
13           month    0.053060
0   Globalstraling    0.035950
3        Lufttrykk    0.024350
4         Vindkast    0.020831
21   Vindretning_x    0.017340
22   Vindretning_y    0.015379
1      Solskinstid    0.012289
15  public_holiday    0.009418
6         d_Friday    0.006057
17          summer    0.003716
7         d_Monday    0.001991
10      d_Thursday    0.001882
11       d_Tuesday    0.001596
12     d_Wednesday    0.001392
18          winter    0.000873
16         raining    0.000638
8       d_Saturday    0.000253
9         d_Sunday    0.000243
```

The model got worse, by about  a 0.5 increase in RMSE!

But looking at the importances, nothing changed! This made me run my base model again, since that is quite interesting that the model is worse but none of the feature importances changed.

I suspect data normalization may be a useful tool sometimes, but in this case it made the model worse, as maybe while data is transformed, it is also lost.  

### TODO - ASK ABOUT THIS

### Changing the n_neighbours for the KNNimputer

Baseline is n_neighbours = 2

After running the model with different n_neighbours, it results in this graph:

| n_neighbours | RMSE   |
| ------------ | ------ |
| 2            | 22.770 |
| 10           | 22.773 |
| 19           | 22.799 |
| 20           | ***22.754*** |
| 21           | 22.799 |
| 23           | 22.778 |
| 25           | 22.815 |
| 30           | 22.804 |

From these attempts, one can see that n_neighbours of 20 results in the lowest RMSE.

Interestingly, when n_neighbours is too high, it results in missing values filled in in a way that makes the model predict traffic values worse, compared to that of when n_neighbours is 20. 

### Removing dummy variables for days

So far, i have taken the dummy variables for days as a given, but what if they actually are making the model worse?
Instead, day will just be a coloumn with a number 0-6

Model for test data = False
MSE: 520.6407280284674
RMSE: 22.817553068382846
```json
           Feature  Importance
13       rush_hour    0.320061
5             hour    0.111321
6              day    0.104902
2   Lufttemperatur    0.104729
8          weekend    0.096677
14       sleeptime    0.067550
7            month    0.053234
0   Globalstraling    0.035788
3        Lufttrykk    0.024492
4         Vindkast    0.020902
15   Vindretning_x    0.017557
16   Vindretning_y    0.015657
1      Solskinstid    0.012397
9   public_holiday    0.009422
11          summer    0.003761
12          winter    0.000888
10         raining    0.000664
```

Removing dummy variables for days made the model worse.
The new best RMSE is 22.754, and removing dummy variables led to an RMSE of 22.81. Looking at previous model importances, the days did not seem to be very important, but trying without the days as dummies did provide insight into their importance. 

### Removing 2020 and 2021

MSE: 528.4341584282549
RMSE: 22.98769580510963

2020 and 2021 were very different years due to the COIVD-19 pandemic. 

Completely removing these years led to a higher RMSE. The reason i bring this up is because if the goal is to predict 2023 data, it may be smart to drop these years since 2023 society resembles 2017-2019 society more than 2020-2021. However, removing so much data would probably do more harm than good, as traffic did not vary that much across these two years.

![FloridaDanmarksplass vs time](src/figs/timeVStrafficBoth.png)

Simply graphing average traffic per hour for each year also reveals that while there is some variance between years, the general idea of a "rush hour" still stands.

Interestingly, the average amount of cycle traffic was slightly higher for 2020:

From the training data: (does not include 2021.)

| Year | Mean Traffic |
|------|--------------|
| 2015 | 50.576    |
| 2016 | 49.743    |
| 2017 | 49.109    |
| 2018 | 47.451    |
| 2019 | 54.560    |
| 2020 | 58.594    |


### Removing the raining coloumn

The idea behind the "raining" coloumn, is that when the air pressure is below 996, it may be a way to indicate raining. 
This idea came from research below:
[Rain air pressure link]("https://geo.libretexts.org/Bookshelves/Oceanography/Oceanography_101_(Miracosta)/08%3A_Atmospheric_Circulation/8.08%3A_How_Does_Air_Pressure_Relate_to_Weather#:~:text=Increasing%20high%20pressure%20(above%201000,corresponds%20with%20cloudy%2C%20rainy%20weather.")

After removing the rain coloumn, RMSE increased to 
RMSE: 22.794. so an increase of 0.04. This proves that this coloumn helped the model, it also implies that the idea of rain appearing below a certain air pressure, but does not actually prove it. It may be a complete coincedence.

### Adding a year coloumn

Model for test data = False
MSE: 652.8840755830721
RMSE: 25.551596341189175
           Feature  Importance
20       rush_hour    0.320061
15         weekend    0.189351
5             hour    0.111178
2   Lufttemperatur    0.101366
21       sleeptime    0.067550
14           month    0.052892
0   Globalstraling    0.033611
3        Lufttrykk    0.022354
13            year    0.018281
4         Vindkast    0.017599
22   Vindretning_x    0.014814
23   Vindretning_y    0.013099
1      Solskinstid    0.011184
16  public_holiday    0.009163
6         d_Friday    0.005836
18          summer    0.003750
7         d_Monday    0.001809
10      d_Thursday    0.001655
11       d_Tuesday    0.001340
12     d_Wednesday    0.001198
19          winter    0.000799
17         raining    0.000623
8       d_Saturday    0.000261
9         d_Sunday    0.000228

It got worse! Hmm maybe year is just confusing!




### TEST DATA :

After experimenting and finding the best model for this use case, the model was checked against test data, to see if the model can actually generalize, or if it is just good at the training and valdiation data. 


Model for test data = True
MSE: 570.7803538989668
RMSE: 23.891009897008683
```json
           Feature  Importance
19       rush_hour    0.320061
14         weekend    0.189385
5             hour    0.111220
2   Lufttemperatur    0.104526
20       sleeptime    0.067550
13           month    0.053060
0   Globalstraling    0.035950
3        Lufttrykk    0.024350
4         Vindkast    0.020831
21   Vindretning_x    0.017340
22   Vindretning_y    0.015379
1      Solskinstid    0.012289
15  public_holiday    0.009418
6         d_Friday    0.006057
17          summer    0.003716
7         d_Monday    0.001991
10      d_Thursday    0.001882
11       d_Tuesday    0.001596
12     d_Wednesday    0.001392
18          winter    0.000873
16         raining    0.000638
8       d_Saturday    0.000253
9         d_Sunday    0.000243
```

### Results discussion:

**The chosen best model was "RandomForestRegressor" with an *n_estimators* of 200 #TODO.**

The results present a model which is suprisingly good, considering the amount of variance in the data. 

Simply comparing it to a DummyRegressor, the model is a lot better, as the DummyRegressor gets a RMSE on test data of:
#TODO. 

Looking at the predicted values for 2023, the model picks up on a few key things.

- In the middle of the night (22:00-04:00), traffic drops to 1/2 cyclists.
- During rush hour (07:00-09:00) and (14-17:00) the traffic shoots to 200, and to even higher numbers if the weather is nice. 
- There is a smaller amount of traffic between (09:00-14:00) and (17:00-22:00)

The model seems to get the general gist of what causes cyclist traffic to vary, but predicting the exact values is almost an impossible task. A good way to represent this is graphing the difference between the highest and lowest traffic value for each hour (on training data)

![diff min/max traffic per hour](src/yearfigs/traffic_diff_perhour.png)

This exempliflies how much the traffic varies, and how daunting of a task it would be to guess exact values. 

The amount of variables one could imagine could have an effect on traffic are almost endless. One could imagine a coloumn which was "% of votes for MDG" in the past voting year. This could have an effect on the amount of people cycling, as more people voting "green" could reflect an increasingly cycle-friendly culture. The point is, given the data, i am impressed that the model is this "close" to reality. 
 
The model is not exact, but this is due to the numbers never being "exact" in reality, and an RMSE of around 20 is very reasonable. 


----------------------------

### Model improvements

Given enough time and data, the model could be improved upon in a vriety of ways. Choosing a more complex, model which is made to excel in data over time, could improve the model.
More data such as the actual amount of precipitation, the amount of ice on the ground, the current news scene, or data around COVID-19 restrictions could have made the model better.

I think a RandomForestRegressor with an even higher *n_estimators* past 200 could make the model marginally better aswell.













### Conclusion:

After performing data analysis, feature engineering, and data transformation, this project explored an approach to creating a somewhat accurate model for approximating cycling traffic given the weather conditions. 