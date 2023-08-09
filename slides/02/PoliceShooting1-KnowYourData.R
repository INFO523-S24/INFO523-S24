# INFO 523: PoliceShooting1 Data for Ch2 Know Your Data 
#if you have not used R for a while, you might want to update R 
#Run the following in RGui (not R Studio) to update R
#Rstudio will used the latest R

install.packages("installr")
library(installr)
# updateR()

################################### Get the Data Ready ############################


#Download police shooting data from https://github.com/washingtonpost/data-police-shootings
#Read the dataset description.

#Read csv file into a dataframe
install.packages("readr")
library(readr)
install.packages("tibble")
library(tibble)
install.packages("dplyr")
library(dplyr)

install.packages("tidyverse")
library(tidyverse)

shooting_orig <-read_csv("/cloud/project/fatal-police-shootings-data.csv", col_names = TRUE, na="")
summary(shooting_orig)

getwd()
#setwd("/Users/yanhan/Documents/Info523-SP21/week3/")
setwd(getwd())
getwd()
save(shooting_orig, file="shooting_orig.Rdata")

typeof(shooting_orig)
is.data.frame(shooting_orig)
is_tibble(shooting_orig)

#view the data
View(shooting_orig)
head(shooting_orig)

#obtain a summary of the data: data types of the columns
str(shooting_orig)
summary(shooting_orig)
#from the summary: 1/4 victims had signs of mental illness, 1/9 cases police wore a body camera

#mean, median,mode = 25
mean(shooting_orig$age, na.rm = TRUE)
median(shooting_orig$age, na.rm= TRUE) #positively skewed, or skewed right, meaning younger people has higher density, older people are distributed across a wider range.
sort(table(shooting_orig$age))
#5-number summary
summary(shooting_orig$age)
var(shooting_orig$age, na.rm =TRUE)

######### Plotting #################
#see some skewness (skew right or positively skewed)
boxplot(shooting_orig$age)

#check the normality of age distribution using QQ normal plot: 
#if data were normaly distributed, the age should have been higher than observed
qqnorm(shooting_orig$age)

#histogram
hist(shooting_orig$age, freq=FALSE) #plot density, if plot counts, set freq=TRUE or remove freq)

##################### Take a look at other attributes #################
#want to know unique values in some of the columns
unique(shooting_orig$state)
#All 50 states are covered
#check all variables
apply(shooting_orig, 2, unique)

#distribution in flee
table(shooting_orig$flee)/sum(table(shooting_orig$flee)) #2/3 cases not fleeing

#get a sense of the extend of missing data (lots of values for race is missing)
#815 out of 4478 rows has some missing value
filter(shooting_orig, !complete.cases(shooting_orig))
nrow(shooting_orig)

#apply() returns a vector or array or list of values obtained by applying a 
#function to margins of an array or matrix.
#number of NAs in each column ('2')
apply(shooting_orig, 2, function(x) sum(is.na(x))) %>% sort(decreasing=TRUE) 
#What shall we do with NAs? Not clear at this time, so keep NAs as is. 

#which stae/city has most shooting cases?
sort(table(shooting_orig$state), decreasing =FALSE)
dplyr::count(shooting_orig, city, sort = TRUE)

################## STATES ####################
install.packages("ggplot2")
library(ggplot2)

#shooting distribution by race
plot1 <-ggplot(shooting_orig, aes(x=race)) +
  geom_bar() +
  ggtitle("shooting distribution by race")
plot1

# Race distribution in U.S.A. censu.gov 
pop.race <-c(rep("A",6), rep("B", 13), rep("H", 19), rep("N", 2),  rep("W", 60))
us <- data.frame(pop.race)
plot2 <-ggplot(us, aes(x=pop.race)) +
  geom_bar() + 
  ggtitle("US distribution by race")
plot2

# comparing two plots
install.packages("gridExtra")
library(gridExtra)
grid.arrange(plot1, plot2, ncol=2)

#shooting distribution by states
ggplot(shooting_orig, aes(x=state, fill=state)) +
  geom_bar() +
  ggtitle("shooting distribution by states")

#bring the race factor in
ggplot(shooting_orig, aes(x=state, fill=race)) +
  geom_bar() +
  ggtitle("shooting distribution by states")


#now I see the need to make the x-tick marks more readable
#http://www.sthda.com/english/wiki/ggplot2-axis-ticks-a-guide-to-customize-tick-marks-and-labels
ggplot(shooting_orig, aes(x=state, fill=race)) +
  geom_bar() +
  ggtitle("shooting distribution by states") +
  scale_y_continuous(expand = c(0, 0)) +
  theme(axis.text.x = element_text(size=6, angle=90), 
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        panel.background = element_blank(), 
        axis.line = element_line(colour = "black"))

#Are the number of incidence correlated with state population?
#Bring popluation data in to the picture

install.packages("usmap")
library(usmap)

#usmap data has regions and poverty info we could use
statepop #state population
statepov #state poverty index

#is there a correlation between incident count and population or poverty level of the states?
#find incident count for each of the states
(state_count <- arrange(count(shooting_orig, state), desc(n)))

#create joint dataframe by matching on state abbreviations to run correlation

#need to join on 'state', so change colname 'abbr' used in statepop/statepov to 'state'
colnames(statepop)[colnames(statepop) == "abbr"] <- "state"
colnames(statepov)[colnames(statepov) == "abbr"] <- "state"
#use dplyer inner_join to join state incident count and state population/poverty tables
(state_join_pop <- inner_join(state_count, statepop[, c("state", "pop_2015")], by=c("state")))
(state_join_pov <- inner_join(state_count, statepov[, c("state", "pct_pov_2014")], by=c("state")))

#correlation btw shooting and population size and poverty level
#scatter plot
plot(state_join_pop$n, state_join_pop$pop_2015, main="Relationships btw shooting count and population size")
#correlation coefficience
cor(state_join_pop$n, state_join_pop$pop_2015)
#population size is highly correlated

plot(state_join_pop$n, state_join_pov$pct_pov_2014, main="Relationships btw shooting count and porverty level")
cor(state_join_pop$n, state_join_pov$pct_pov_2014)
#poverty level is not as highly correlated

View(arrange(statepov, desc(pct_pov_2014)))


############### relationship with date?  => year, month
install.packages("lubridate")
library(lubridate)

shooting_orig <- mutate(shooting_orig,
                        year = year(shooting_orig$date),
                        month = month(shooting_orig$date)
)

colnames(shooting_orig)
shooting_orig[, c('date', 'year', 'month')]

ggplot(shooting_orig, aes(x=month)) +
  geom_bar() +
  ggtitle("shooting distribution by month")
#winter months have fewer shootings, why? Sept - Dec

ggplot(shooting_orig, aes(x=month)) +
  geom_bar() + facet_grid("year~race")
ggtitle("shooting distribution by month")

