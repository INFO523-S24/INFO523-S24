# INFO 523: PoliceShooting1 Data for Ch3: Data Preprocessing

################################### Get the Data Ready ############################

# Download police shooting data from https://github.com/washingtonpost/data-police-shootings
# Read the dataset description.
#
install.packages("dplyr")
library(dplyr)
install.packages("ggplot2")
library(ggplot2)

# Load R data
setwd(getwd())
getwd()
load(file="shooting_orig.Rdata")

#view the data
str(shooting_orig)

# #examining missing data
# #get a sense of the extend of missing data (lots of values for race is missing)
# #805 out of 4478 rows has some missing value
filter(shooting_orig, !complete.cases(shooting_orig))
 
# #apply() returns a vector or array or list of values obtained by applying a function to margins of an array or matrix.
apply(shooting_orig, 2, function(x) sum(is.na(x))) %>% sort(decreasing=TRUE) #number of NAs in each column ('2')
###### What shall we do with NAs? Not clear at this time, so keep NAs as is. 

###############
#Need to bin age to several bins
#plot age
ggplot(shooting_orig, aes(x=age, fill=age)) +
  geom_histogram() +
  ggtitle("Shooting Distribution by age")
boxplot(shooting_orig$age)
summary(shooting_orig$age)

# could use this to pick a better binwidth for the histogram. Where are the cut points?
# W=2?IQR?n^(???1/3) . N = range/W

IQR(shooting_orig$age, na.rm=TRUE) #IQR is Q3-Q1, middle 50%. 
(width <- 2*IQR(shooting_orig$age, na.rm=TRUE)*(nrow(shooting_orig[!is.na(shooting_orig$age),]))^(-1/3))
(N <- (max(shooting_orig$age, na.rm=TRUE) - min(shooting_orig$age, na.rm = TRUE))/width)
#N too large, good for histogram/data smoothing, not for discretization to reduce cardinality

# method1: binning with fixed equal-width cut
# <25, 25-50, 50-75, >75
# brains are not fully developed before 25. 
(ageGroup <- cut(shooting_orig$age, 
  breaks = c(0, 25, 50, 75, 100), labels=c('young', 'grown', 'mature', 'old')))

# method2: binning by k-means clustering
age <- shooting_orig$age[!is.na(shooting_orig$age)]
k<- kmeans(age, centers = 4)   # 4 clusters
table(k$cluster)
k$centers

# running k-means the 2nd time. notice the differences of each run. results are different. why?
k<- kmeans(age, centers = 4)   # 4 clusters
table(k$cluster)
k$centers

plot(age, col=k$cluster)
# sum of squares from points to the assigned cluster centers is minimized. 
age_cluster <- data.frame(age=age, cluster=k$cluster)

# look at all 4 clusters
for(i in 1:4){ 
  cat("cluster", i, ":[", min(age_cluster[age_cluster$cluster==i,"age"]), ",",
      max(age_cluster[age_cluster$cluster==i,"age"]), "]\n")
}

# method3: binning with decision tree (finding the best cuts that correlated with label:race)
# use age to predict race
install.packages("rpart") # rpart: recursive partitioning and regression trees
library(rpart) 

age_tree <- rpart(
  formula = race ~ age,
  data    = shooting_orig,
  parms   = list(split="information"),
  maxdepth= 3,
  method  = "class"
)
library(rpart.plot)
prp(age_tree, type=4, extra=109) # https://www.rdocumentation.org/packages/rpart.plot/versions/3.1.1/topics/prp 
# cuts: 0-25, 26-43, >44 


# method4:ChiMerge
# Package 'discretization': 2022. https://cran.r-project.org/web/packages/discretization/discretization.pdf
install.packages("discretization")
library("discretization")
ar <- shooting_orig[!is.na(shooting_orig$race) & !is.na(shooting_orig$age) , c("age","race")]
data <- data.frame(age=ar$age, race=ar$race) #chiM doesn't work on ar
age_chi = chiM(data, alpha=0.05) #assume the last column is the class label
age_chi$cutp # show cut points
table(age_chi$Disc.data$age)

# use decision tree cuts 
# shooting_orig$ageGroupDT <- cut(shooting_orig$age, breaks = c(0, 26, 43, 100), labels=c('young', 'grown', 'old'))
# dplyr::glimpse(shooting_orig)

# Use ChiMerge
shooting_orig$ageGroupDT <- cut(shooting_orig$age, 
                              breaks = c(0, 22.5, 25.5, 43.5, 47.5, 100), 
                              labels=c('Adolescent','Young Adult', 'Adult', 'Middle-aged', 'Adult-Senior'))
dplyr::glimpse(shooting_orig)

################## ARMED  ###############
unique(shooting_orig$armed)
# 85 different values, need to combine them into fewer types
# keep the original armed column, add armedType column.

# firearms and objects look like a gun
# knives
# vehicles
# other objects
# unarmed
# undetermined

# options(tibble.print_max = Inf) #to print an entire tibble
count(shooting_orig, armed, sort=TRUE)
# create a function called group(). Could have used fct_collapse() but the function was buggy (issue reported and fixed) 
group <- function (string){
  if(is.na(string)) return ("NA")
  if(string == "unarmed") return ("unarmed")
  else if (string == "undetermined") return ("undetermined")
  else if (string == "vehicle") return ("vehicle")
  else if (string %in% c("gun", "toy weapon", "gun and knife", "gun and car", "BB gun", "guns and explosives", "gun and vehicle", "hatchet and gun", "gun and sword", "machete and gun", "vehicle and gun", "pellet gun"))  return ("gun")
  else if (string %in% c("knife", "ax", "sword", "box cutter", "hatchet", "sharp object", "scissors", "meat cleaver", "pick-axe", "straight edge razor", "pitchfork", "chainsaw", "samurai sword", "spear")) return ("sharpObject")
  else return ("other")
}

shooting_orig$armedType <- sapply(shooting_orig$armed, group)
dplyr::glimpse(shooting_orig)

# What other attributes have high cardinalities? 
apply(shooting_orig, 2, unique) #show unique values for each of the columns
# some machine learning algorithms e.g. decision tree, don't handle high cardinality data well (very slow)

# states and cities. what can we do with them?

#### States
### 51 states is too many states for decision trees
### may be grouped into regions, or by poverty level
install.packages("usmap")
library(usmap)
#usmap data has regions and poverty info we could use
statepop #state population
statepov #state poverty index

# Is there a correlation between incident count and population or poverty level of the states?
# find incident count for each of the states
(state_count <- arrange(count(shooting_orig, state), desc(n)))

# create joint dataframe to run correlation

# need to join on 'state', so change colname 'abbr' used in statepop/statepov to 'state'
colnames(statepop)[colnames(statepop) == "abbr"] <- "state"
colnames(statepov)[colnames(statepov) == "abbr"] <- "state"
# use dplyer inner_join to join state incident count and state population/poverty tables
(state_join_pop <- inner_join(state_count, statepop[, c("state", "pop_2015")], by=c("state")))
(state_join_pov <- inner_join(state_count, statepov[, c("state", "pct_pov_2014")], by=c("state")))

# correlation btw shooting and population size and poverty level
plot(state_join_pop$n, state_join_pop$pop_2015, main="Relationships btw shooting count and population size")
cor(state_join_pop$n, state_join_pop$pop_2015)
# population size is highly correlated

plot(state_join_pop$n, state_join_pov$pct_pov_2014, main="Relationships btw shooting count and porverty level")
cor(state_join_pop$n, state_join_pov$pct_pov_2014)
# poverty level is not correlated

# View(arrange(statepov, desc(pct_pov_2014)))

# add a new attribute victim/population ratio (vpratio)
state_join_pop$vpRatio <- round(scale(state_join_pop$n/state_join_pop$pop_2015), 2)

# discretize pov and pop, and then add new features to shooting data for future use
state_join_pov$povLevel<- cut(state_join_pov$pct_pov_2014, breaks=5, labels = c("VeryLowPov", "LowPov", "MedPov", "MedHigPov", "HighPov")) #equal width
state_join_pop$popLevel<- cut(state_join_pop$pop_2015, breaks=5, labels = c("VeryLowPop", "LowPop", "MedPop", "MedHigPop", "HighPop")) #equal width
state_join_pop$vpRatioLevel<- cut(state_join_pop$vpRatio, breaks=5, labels = c("VeryLowVPRatio", "LowVPRatio", "MedVPRatio", "MedHigVPRatio", "HighVPRatio")) #equal width

# or use clustering
#kp<- kmeans(state_join_pov$pct_pov_2014, centers = 5)
#table(kp$cluster)
#kp$centers
#plot(state_join_pov$pct_pov_2014, col=kp$cluster)
# compare
#table(cut(state_join_pov$pct_pov_2014, breaks=5))

#join with shooting_orig
shooting_orig <- inner_join(shooting_orig, state_join_pop[, c("state", "popLevel", "vpRatio", "vpRatioLevel")], by="state") %>%
  inner_join(state_join_pov[, c("state", "povLevel")], by="state")

colnames(shooting_orig)

#date => year, season
install.packages("lubridate")
library(lubridate)

shooting_orig <- shooting_orig %>% mutate(
  year = year(shooting_orig$date),
  month = month(shooting_orig$date)
)

colnames(shooting_orig)
shooting_orig[, c('date', 'year', 'month')]

setwd(getwd())
getwd()
shooting <- shooting_orig
save(shooting, file="shooting.RData")

  