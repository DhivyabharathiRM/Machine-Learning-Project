# Species Prediction using K-Means Clustering

# Importing Library
library(ggpubr)
library(factoextra)
library(cluster)
library(fpc)
library(VIM)

# Analyzing Dataset
data('iris')
df = iris
head(df,5)
summary(df)

# Visualizing Missing Data
aggr(df)

# Feature Selection
x = df[,-5]
x

# Elbo Method to find number of clusters
set.seed(6)
wcss = vector()
for (i in 1:10) wcss[i] = sum(kmeans(x,i)$withinss)
plot(1:10, wcss, type = 'b', main = paste('Clusters of Species'),
     xlab = 'number of clusters', ylab = 'wcss')

# Apllying K-Means to Iris Dataset
set.seed(29)
km = kmeans(x, 3, nstart = 10)

# Pair Plot
with(df, pairs(x, col=c(1:3)[km$cluster]))

# Cluster Plot
plotcluster(x,km$cluster)

fviz_cluster(km, data = x,
             palette = c('#2E9FDF','#00AFBB','#E7B800'),
             geom = 'point',
             ellipse.type = 'convex',
             ggtheme = theme_bw())

# Cluster Plot with Centroid
res.pca = prcomp(x,scale. = TRUE)
ind.coord = as.data.frame(get_pca_ind(res.pca)$coord)
ind.coord$cluster = factor(km$cluster)
ind.coord$Species  = df$Species
head(ind.coord)

eigenvalue = round(get_eigenvalue(res.pca),1)
variance.percent = eigenvalue$variance.percent
head(eigenvalue)

ggscatter(ind.coord, x = 'Dim.1', y = 'Dim.2', color = 'cluster',
          palette = 'npg', ellipse = TRUE, ellipse.type = 'convex',
          shape = 'Species', size = 1.5, legend = 'right', 
          ggtheme = theme_bw(),
          xlab = paste0('Dim 1 (',variance.percent[1],'%)'),
          ylab = paste0('Dim 2 (',variance.percent[2],'%)')
)+ stat_mean(aes(color = cluster), size = 4)
