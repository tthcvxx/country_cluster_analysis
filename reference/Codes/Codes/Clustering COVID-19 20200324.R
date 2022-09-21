
#https://www.r-spatial.org/r/2018/10/25/ggplot2-sf.html
rm(list = ls(all.names = TRUE))
options(max.print=1000000)
library("rnaturalearth")
library("rnaturalearthdata")
library("ggplot2")
library(gridExtra)



### READ DATA

data <- read.csv("C:/Users/rcarrill/Box Sync/COVID/03. Clustering/_Data/data_2020-03-24.csv")             # most recent surveillance
data.date <- read.csv("C:/Users/rcarrill/Box Sync/COVID/03. Clustering/_Data/data_date_2020-03-24.csv")   # most recent surveillance in long format
data.date <- subset(data.date, value > 0)
data.date <- data.date[which(data.date$date == "2020-03-21"),]
w6.cluster <- read.csv("C:/Users/rcarrill/Box Sync/COVID/03. Clustering/_Data/Dataset_6_clusters.csv")        # cluster database
w5.cluster <- read.csv("C:/Users/rcarrill/Box Sync/COVID/03. Clustering/_Data/Dataset_5_clusters.csv")        # cluster database


world <- ne_countries(scale = "medium", returnclass = "sf")
world$name[which(world$name == "Antigua and Barb.")] <- "Antigua and Barbuda"
world$name[which(world$name == "Bosnia and Herz.")] <- "Bosnia and Herzegovina"
world$name[which(world$name == "Russia")] <- "Russian Federation"
world$name[which(world$name == "Dominican Rep.")] <- "Dominican Republic"
world$name[which(world$name == "Côte d'Ivoire")] <- "Cote d'Ivoire"
world$name[which(world$name == "St. Vin. and Gren.")] <- "Saint Vincent and the Grenadines"
world$name[which(world$name == "Korea")] <- "South Korea"
world$name[which(world$name == "Czech Rep.")] <- "Czech Republic"
world$name[which(world$name == "Dem. Rep. Congo")] <- "Congo (Kinshasa)"
world$name[which(world$name == "Congo")] <- "Congo (Brazzaville)"
world$name[which(world$name == "Eq. Guinea")] <- "Equatorial Guinea"
world$name[which(world$name == "Central African Rep.")] <- "Central African Republic"
world$name[which(world$name == "Taiwan")] <- "Taiwan (Province of China)"
world$name[which(world$name == "Gambia")] <- "The Gambia"
world$name[which(world$name == "Bahamas")] <- "The Bahamas"





### PLOT WORD MAPS

world.w5 <- merge(world, w5.cluster, by.x = c("name") , by.y = c("Country"), all.x=T)
world.w6 <- merge(world, w6.cluster, by.x = c("name") , by.y = c("Country"), all.x=T)

print(w5.cluster$Country[!w5.cluster$Country%in%world.w5$name])    # CHECK THAT ALL COUNTRIES WERE MERGED
print(w6.cluster$Country[!w6.cluster$Country%in%world.w6$name])


#1000x500 pixeles
ggplot(data = world.w5) +
  geom_sf(aes(fill = as.factor(label))) +
  theme_bw() +
  scale_fill_discrete(name = "Clusters", na.translate=F) 


ggplot(data = world.w6) +
  geom_sf(aes(fill = as.factor(label))) +
  theme_bw() +
  scale_fill_discrete(name = "Clusters", na.translate=F) 





### CLUSTERS COMPARISON -> EVENT DATA UNTIL 23/03/2020!!!!!!

data$Country <- as.character(data$Country)
data$Country[which(data$Country == "Bahamas")] <- "The Bahamas"
data$Country[which(data$Country == "Gambia")] <- "The Gambia"


data <- merge(data, w5.cluster[,c(4,5)], by = c("Country"), all.x = T)
names(data)[names(data) == 'label'] <- 'label5'
data <- merge(data, w6.cluster[,c(4,5)], by = c("Country"), all.x = T)
names(data)[names(data) == 'label'] <- 'label6'
data <- merge(data, data.date, by.x = c("Country"), by.y = c("Country.Region"), all.x = T)


print(w5.cluster$Country[!w5.cluster$Country%in%data$Country])
print(w6.cluster$Country[!w6.cluster$Country%in%data$Country])


data$death_rate <- 1000*(data$total_deaths/data$value)


table(data$Country, data$label5, useNA = c("always"))    # CHECK FOR MISSING COUNTRIES
table(data$Country, data$label6, useNA = c("always"))


f.c5.value <- ggplot(data[which(!is.na(data$label5)),], aes(x=as.factor(label5), y=log10(value)))+
              geom_boxplot() +
              theme_bw() +
              xlab("Cluster") + ylab("Log10-Confirmed cases")

f.c6.value <- ggplot(data[which(!is.na(data$label6)),], aes(x=as.factor(label6), y=log10(value)))+
              geom_boxplot() +
              theme_bw() +
              xlab("Cluster") + ylab("Log10-Confirmed cases")
              


f.c5.deaths <- ggplot(data[which(!is.na(data$label5)),], aes(x=as.factor(label5), y=log10(total_deaths)))+
               geom_boxplot() +
               theme_bw() +
               xlab("Cluster") + ylab("Log10-Deaths") +
               ylim(0, 5)

f.c6.deaths <- ggplot(data[which(!is.na(data$label6)),], aes(x=as.factor(label6), y=log10(total_deaths)))+
               geom_boxplot() +
               theme_bw() +
               xlab("Cluster") + ylab("Log10-Deaths") +
               ylim(0, 5)



f.c5.death.rate <- ggplot(data[which(!is.na(data$label5)),], aes(x=as.factor(label5), y=log10(death_rate)))+
                   geom_boxplot() +
                   theme_bw() +
                   xlab("Cluster") + ylab("Log10-Death rates") +
                   ylim(0, 5)

f.c6.death.rate <- ggplot(data[which(!is.na(data$label6)),], aes(x=as.factor(label6), y=log10(death_rate)))+
                   geom_boxplot() +
                   theme_bw() +
                   xlab("Cluster") + ylab("Log10-Death rates") +
                   ylim(0, 5)



f.c5.order <- ggplot(data[which(!is.na(data$label5)),], aes(x=as.factor(label5), y=order))+
              geom_boxplot() +
              theme_bw() +
              xlab("Cluster") + ylab("Order") +
              ylim(0, 80)

f.c6.order <- ggplot(data[which(!is.na(data$label6)),], aes(x=as.factor(label6), y=order))+
              geom_boxplot() +
              theme_bw() +
              xlab("Cluster") + ylab("Order") +
              ylim(0, 80)



grid.arrange(f.c5.value, f.c5.deaths, f.c5.death.rate, f.c5.order, 
             f.c6.value, f.c6.deaths, f.c6.death.rate, f.c6.order, 
             nrow = 2)
#1000x500 pixeles


summary(aov(log10(value)~as.factor(label5), data=data))
  with(data, pairwise.t.test(log10(value), as.factor(label5), p.adjust.method = "bonf"))
summary(aov(log10(value)~as.factor(label6), data=data))
  with(data, pairwise.t.test(log10(value), as.factor(label6), p.adjust.method = "bonf"))

  

summary(aov((total_deaths)~as.factor(label5), data=data))
  with(data, pairwise.t.test((total_deaths), as.factor(label5), p.adjust.method = "bonf"))
summary(aov((total_deaths)~as.factor(label6), data=data))
  with(data, pairwise.t.test((total_deaths), as.factor(label6), p.adjust.method = "bonf"))
  

  
summary(aov((death_rate)~as.factor(label5), data=data))
  with(data, pairwise.t.test((death_rate), as.factor(label5), p.adjust.method = "bonf"))
summary(aov((death_rate)~as.factor(label6), data=data))
  with(data, pairwise.t.test((death_rate), as.factor(label6), p.adjust.method = "bonf"))
  

  
summary(aov((order)~as.factor(label5), data=data))
  with(data, pairwise.t.test((order), as.factor(label5), p.adjust.method = "bonf"))
summary(aov((order)~as.factor(label6), data=data))
  with(data, pairwise.t.test((order), as.factor(label6), p.adjust.method = "bonf"))

  
  
### SUPPLEMENTARY MATERIAL
data.sup <- data[,c("Country", "order", "total_deaths", "value", "death_rate", "label5", "label6")]
head(data.sup)
data.sup <- data.sup[which(!is.na(data.sup$label5)),]
write.csv(data.sup, "C:/Users/rcarrill/Box Sync/COVID/03. Clustering/Paper/3.Supplementary Material/Supplementary Material.csv", row.names = F)
