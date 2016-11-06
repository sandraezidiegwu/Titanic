library(rpart) #classification algorithm
library(rpart.plot) #visualization
library(ggplot2) #visualization
library(dplyr) #data manipulation
setwd("/Users/sandraezidiegwu/Documents/Coursera/Data Science/Titanic/")

#Data Collection and Gathering
genderclass <- read.csv("genderclassmodel.csv", header = T, sep = ",")
gender <- read.csv("gendermodel.csv", header = T, sep = ",")
test <- read.csv("test.csv", header = T, sep = ",")
train <- read.csv("train.csv", header = T, sep = ",")

#Date Exploration
names(test)
names(train)
test$Survived <- 0
all.pass <- rbind(train, test)
str(all.pass)
summary(all.pass)

### VARIABLE DESCRIPTIONS:
### survival        Survival
###  (0 = No; 1 = Yes)
### pclass          Passenger Class
### (1 = 1st; 2 = 2nd; 3 = 3rd)
### name            Name
### sex             Sex
### age             Age
### sibsp           Number of Siblings/Spouses Aboard
### parch           Number of Parents/Children Aboard
### ticket          Ticket Number
### fare            Passenger Fare
### cabin           Cabin
### embarked        Port of Embarkation
### (C = Cherbourg; Q = Queenstown; S = Southampton)

# Dig dipper into columns that might affect survival rate. Eg. Title, Family Size, Age etc.
all.pass$Title <- gsub('(.*, )|(\\..*)', '', all.pass$Name)
table(all.pass$Sex, all.pass$Title)

all.pass$Title[all.pass$Title %in% c('Mlle', 'Ms')] <- 'Miss'
all.pass$Title[all.pass$Title %in% c('Dona', 'the Countess', 'Lady', 'Don','Jonkheer', 'Major', 'Sir')] <- 'Affluent'
all.pass$Title[all.pass$Title %in% c('Master', 'Rev', 'Capt', 'Col')] <- 'Mr'
all.pass$Title[all.pass$Title %in% 'Mme'] <- 'Mrs'
table(all.pass$Sex, all.pass$Title)

ggplot(all.pass, aes(x = Title, fill = factor(Survived))) +
geom_bar(stat = 'count', position = 'dodge') 

# Create Family Size column
all.pass$FSize <- all.pass$SibSp + all.pass$Parch + 1

# Vizualize the relationship between family size and survival
ggplot(all.pass, aes(x = FSize, fill = factor(Survived))) + 
  geom_bar(stat = 'count', position = 'dodge') +
  scale_x_continuous(breaks = c(1:11)) +
  labs(x = 'Family Size') +
  theme_linedraw()

# The barplot shows that the larger the family size, the lower the
# chances of survival. I'm also assuming that affluent families had
# a lower family size count. Lets check that... 

ggplot(all.pass, aes(x = FSize, fill = factor(Title))) + 
  geom_bar(stat = 'count', position = 'dodge') +
  scale_x_continuous(breaks = c(1:11)) +
  labs(x = 'Family Size') +
  theme_linedraw()

# I was right! The largest family size for affluent families is 2 so
# they had a good chance of survival. Let's vsualize this further...

par(mfrow = c(1,1))
mosaicplot(table(all.pass$Title, all.pass$Survived), main = 'Survival by Title', shade = TRUE)

# You can see that the 'affluent' folks pretty much survived.. Not sure about
# the Capt. though, according to the movie, he sank with the ship.. But
# hey! What do I know...

# Missing Values
sum(is.na(all.pass$Fare))
na.fare <- all.pass[is.na(all.pass$Fare),]
na.fare

## PassengerId 1044 has an NA Fare value. He/She ranks in 3rd class and embarked form 'S' however.
## We will replace row 1044 fare value with the median fare value of that class and embarkment.
all.pass$Fare[1044] <- median(all.pass[all.pass$Pclass == '3' & all.pass$Embarked == 'S',]$Fare, na.rm = T)

par(mfrow = c(1,1))
hist(all.pass$Age, main = 'Age Distribution aboard the Titanic', xlab = "Age", col = 'lightblue')
sum(is.na(all.pass$Age))

actual <- all.pass
actual2 <- all.pass
age.part <- rpart(Age ~ Pclass+Sex+SibSp+Parch+Ticket+Fare+Cabin+FSize+Title, data = all.pass[!is.na(all.pass$Age),], method = 'anova')
age.pred <- predict(age.part, all.pass[is.na(all.pass$Age),])
actual$Age[is.na(actual$Age)] <- age.pred

# Prune variables to achieve better prediction
age.prune <- prune.rpart(age.part, cp = 0.1)
age.pred2 <- predict(age.prune, all.pass[is.na(all.pass$Age),])
actual2$Age[is.na(actual2$Age)] <- age.pred2

# 20% improvement on error with pruned prediction
mean(actual2$Age != actual$Age)

# Visualize age distributions of the actual data and the predicted values
par(mfrow = c(1,3))
hist(all.pass$Age, col = 'blue', main = 'Actual Age Values', xlab = 'Age')
hist(actual$Age, col = 'lightblue', main = 'Predicted Age Values', xlab = 'Age')
hist(actual2$Age, col = 'lightblue', main = 'Pruned Predicted Age Values', xlab = 'Age')

# Replace missing age values with prune prediction age values
all.pass$Age[is.na(all.pass$Age)] <- age.pred2
sum(is.na(all.pass$Age))
all.pass$AgeDist[all.pass$Age < 18] <- "Child"
all.pass$AgeDist[all.pass$Age >= 18] <- "Adult"
all.pass$AgeDist <- factor(all.pass$AgeDist)

## Let's evaluate survival count by age
table(all.pass$AgeDist, all.pass$Survived)

# Let's move on to predict the survival
## We start by creating test and training data sets
train_data <- all.pass[1:891,]
test_data <- all.pass[892:1309,]
test_data$Survived <- NULL

# We are ready to build our model using random Forest!
model_part <- rpart(Survived ~ Pclass + Sex + Age + SibSp +
                             Parch + Fare + Title + FSize, 
                    data = train_data, method = 'class', control = rpart.control(cp=0.0001))
# Visualize decision tree
par(mfrow=c(1,1))
rpart.plot(model_part)

# Let's predict the test data set
model_pred <- predict(model_part, test_data, type = 'class')

solution <- data.frame(PassengerId = test_data$PassengerId, Survived = model_pred)
solution[1:10,]

# A little prune never hurt nobody. Let's avoid overfitting this model
#```{r}
model_prune <- prune(model_part, cp = 0.1)
model_prune.pred <- predict(model_prune, test_data, type = 'class')
prune_solution <- data.frame(PassengerID = test_data$PassengerId, Survived = model_prune.pred)
prune_solution[1:10,]
#```

# A 10% improvement... not bad!
mean(solution$Survived != prune_solution$Survived)

# Write solution to csv file
write.csv(prune_solution, file = 'rpart-solution.csv', row.names = F)
