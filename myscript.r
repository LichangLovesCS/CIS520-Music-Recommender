# ===============================================================
# read data & preprocessing
#================================================================

df = read.delim(file = 'C:/Users/Yixuan/Documents/000-Semester/CIS-520/project/my_dat.csv', sep = ',', header=FALSE)
colnames(df) <- c("user", "artist", "song")
df <- df[ -c(4:8) ]
df <- df[-which(df$artist == ""), ]
df <- df[-which(df$song == ""), ]

df$A = match(df$artist,levels(df$artist))
df$S = match(df$song,levels(df$song))
df$U = match(df$user,levels(df$user))

df <- df[ -c(1:6) ]

# basic stats
m = nrow(df)
mU = length(unique(df$U))
mG = 12
mS = length(unique(df$S))
mA = length(unique(df$A))

# we are working on smaller sets
# delete unpopular songs that have cumulative <200 plays
songlist = c()
for (i in 1:mS){
  if(sum(df$S==i)>200){
    print(i)
    songlist = c(songlist,i)
  }
}
sdf = df[1,]
for (i in songlist){
  sdf = rbind(sdf,df[df$S==i,])
}
# reorder the factors
sdf$A = as.factor(sdf$A)
sdf$A = match(sdf$A,levels(sdf$A))
sdf$S = as.factor(sdf$S)
sdf$S = match(sdf$S,levels(sdf$S))
sdf$U = as.factor(sdf$U)
sdf$U = match(sdf$U,levels(sdf$U))

# create frequency column to group redundant rows
library(plyr)
sdf=ddply(sdf,.(U,S,A),nrow)
colnames(sdf) <- c("U", "S", "A","F")

# update basic stats
m = nrow(sdf)
mU = length(unique(sdf$U))
mG = 12
mS = length(unique(sdf$S))
mA = length(unique(sdf$A))

# random sample from this smaller set to form our training set
mtr=300
mtst=m-mtr
ind = sample.int(m,mtr)
trsdf = sdf[ind,]
tstsdf = sdf[-ind,]

# Now we have smaller data-set sdf, split into trsdf & tstsdf

#=======================================================================
# Learning Model: E-M
# ======================================================================

# random init params
b = matrix(rexp(mU*mG), ncol=mG)
g = matrix(rexp(mG*mS), ncol=mS)
q = matrix(rexp(mtr*mG),  ncol=mG)

# E-M iterations
logl_curve = c()
for (iter in 1: 550){
  # update: E-step
  for (i in 1:mtr){
    for (j in 1:mG){
      q[i,j] = b[trsdf$U[i],j] / sum(b[trsdf$U[i],]) * g[j, trsdf$S[i]] / sum(g[j,])
    }
    q[i,] = q[i,] / sum(q[i,])
  }
  
  # update: M-step
  # update b and g
  for (i in 1:mtr){
    b[trsdf$U[i],] = b[trsdf$U[i],] + q[i,] * trsdf$F[i]
    g[,trsdf$S[i]] = g[,trsdf$S[i]] + t(q[i,]) * trsdf$F[i]
  }
  
  # expected complete data log likelihood
  L = 0
  for (i in 1:mtr){
    for (j in 1:mG){
      L = L + q[i,j]* ( log(b[trsdf$U[i],j] / sum(b[trsdf$U[i],])) + log(g[j, trsdf$S[i]] / sum(g[j,])) )
    }
  }
  print(L)
  logl_curve = c(logl_curve,L)
}


#==================================================================
# performance evaluation
#==================================================================

# generate 10000-song play list for each user
num_song = 10000
playlist = matrix(nrow=mU,ncol=num_song)
for (i in 1:mU){
  for (k in 1:num_song){
    genre = match(1,rmultinom(1,1,b[i,]))
    song = match(1,rmultinom(1,1,g[genre,]))
    playlist[i,k] = song
  }
}

# Evaluate each metric

rank_results = c()
rank_results_w = c()
freq_pred = c()
freq_gt = c()

for (i in unique(tstsdf$U)){
  user_list = as.data.frame(table(playlist[i,]))
  user_list = user_list[order(-user_list$Freq),]
  
  # measure rank performance
  user_tst = tstsdf[tstsdf$U==i,]
  for (j in user_tst$S){
    rank_results = c(rank_results, match(j,user_list$Var1))
    rank_results_w = c(rank_results_w, user_tst[user_tst$S==j,4])
  }
  
  # measure freq prediction performance
  user_tr = trsdf[trsdf$U==i,]
  total_freq = sum(user_tr$F)
  playlist_old_freq = 0
  for (j in user_tr$S){
    playlist_old_freq = playlist_old_freq + user_list[user_list$Var1==j,2]
  }
  scale_up = total_freq / playlist_old_freq
  user_list$Adj.Freq = user_list$Freq * scale_up
  for (j in user_tst$S){
    if( sum(user_list$Var1==j) > 0 ) {
      freq_gt = c(freq_gt, user_tst[user_tst$S==j,4])
      freq_pred = c(freq_pred, user_list[user_list$Var1==j,3])
    }
  }

}

rank_results_pure = rank_results
rank_results[is.na(rank_results)] = 109
hist(rank_results)
median(rank_results)


#==================================================================
# Learning Model: Gibbs Sampler
#==================================================================

# expand training set (replicate entries according to frequency)
trsdf_exp = data.frame(U=integer(), S=integer(), A=integer(), G=integer())
for (i in 1:nrow(trsdf)){
  print(i)
  for (k in 1:trsdf[i,4]){
    trsdf_exp = rbind(trsdf_exp,trsdf[i,1:3])
  }
}

#trsdf_exp$G = 0

# update basic stats
mtr_exp = nrow(trsdf_exp)
num_samples = 5000
mU = 44 
# User 45 is not in training set but in test set. 
# For simplicity ignore users 45-50 for training and testing.

# random init params
someData <- rexp(num_samples*mU*mG);  
b = array(someData, c(num_samples, mU, mG));  
someData <- rexp(num_samples*mU*mG);  
g = array(someData, c(num_samples, mG, mS)); 
someData <- rexp(num_samples*mtr_exp);  
genre = array(someData, c(num_samples, mtr_exp)); 

# gibbs iterations
for (iter in 2: num_samples){
  print(iter)
  # sample genre
  for (i in 1:mU){
    # compute genre multinom dist for each user
    prob = rep(0,mG)
    for (j in 1:mG){
      prob[j] = b[iter-1,trsdf_exp$U[i],j] / sum(b[iter-1,trsdf_exp$U[i],]) * g[iter-1,j, trsdf_exp$S[i]] / sum(g[iter-1,j,])
    }
    prob = prob / sum(prob)
    
    num_entries = nrow(trsdf_exp[trsdf_exp$U==i,])
    temp = rmultinom(num_entries,1,prob)
    temp = cbind(1:nrow(t(temp)), max.col(t(temp), 'first'))
    trsdf_exp$G[trsdf_exp$U==i] = temp[,2]
  }
  genre[iter,] = trsdf_exp$G
  
  library(MCMCpack)
  # sample b
  for (i in 1:mU){
    aUK = rep(0,mG)
    for (j in 1:mG){
      aUK[j] = 1 + sum(trsdf_exp$U==i & trsdf_exp$G==j)
    }
    b[iter,i,] = rdirichlet(1,aUK)
  }
  
  # sample g
  for (i in 1:mG){
    aGg = rep(0,mS)
    for (j in 1:mS){
      aGg[j] = 1 + sum(trsdf_exp$G == i & trsdf_exp$S == j)
    }
    g[iter,i,] = rdirichlet(1,aGg)
  }
}

# acf shows that we need to give up one sample every 200 iterations
# burn-out is small from observation. we will exclude first 1000.
acf_gap = 200
burnout = 1000
num_true_samples = (num_samples - burnout) / acf_gap

# Get true samples after account for acf and burn-out
someData <- rexp(num_true_samples*mU*mG);  
b_tr = array(someData, c(num_true_samples, mU, mG));  
someData <- rexp(num_true_samples*mU*mG);  
g_tr = array(someData, c(num_true_samples, mG, mS)); 
someData <- rexp(num_true_samples*mtr_exp);  
genre_tr = array(someData, c(num_true_samples, mtr_exp)); 

for (i in 1:num_true_samples){
  idx = burnout+acf_gap*i
  b_tr[i,,] = b[idx,,]
  g_tr[i,,] = g[idx,,]
  genre_tr[i,] = genre[idx,]
}



#==================================================================
# performance evaluation
#==================================================================

# generate 10000-song play list for each user
num_song_per_param = 10000 / num_true_samples
playlist = matrix(nrow=mU,ncol=num_song)
for (i in 1:mU){
  for (k in 1:num_true_samples){
    for (j in 1:num_song_per_param){
      genre = match(1,rmultinom(1,1,b_tr[k,i,]))
      song = match(1,rmultinom(1,1,g_tr[k,genre,]))
      playlist[i,(k-1)*num_song_per_param+j] = song
    }
  }
}

# Evaluate each metric

rank_results = c()
rank_results_w = c()
freq_pred = c()
freq_gt = c()

for (i in 1:mU){
  #if (sum(tstsdf$U==i)==0 || sum(trsdf$U==i)==0){
  #  continue
  #}
  user_list = as.data.frame(table(playlist[i,]))
  user_list = user_list[order(-user_list$Freq),]
  
  # measure rank performance
  user_tst = tstsdf[tstsdf$U==i,]
  for (j in user_tst$S){
    rank_results = c(rank_results, match(j,user_list$Var1))
    rank_results_w = c(rank_results_w, user_tst[user_tst$S==j,4])
  }
  
  # measure freq prediction performance
  user_tr = trsdf[trsdf$U==i,]
  total_freq = sum(user_tr$F)
  playlist_old_freq = 0
  for (j in user_tr$S){
    playlist_old_freq = playlist_old_freq + user_list[user_list$Var1==j,2]
  }
  scale_up = total_freq / playlist_old_freq
  user_list$Adj.Freq = user_list$Freq * scale_up
  for (j in user_tst$S){
    if( sum(user_list$Var1==j) > 0 ) {
      freq_gt = c(freq_gt, user_tst[user_tst$S==j,4])
      freq_pred = c(freq_pred, user_list[user_list$Var1==j,3])
    }
  }
  
}

rank_results[is.na(rank_results)] = 109
hist(rank_results)
plot(log(freq_gt),log(freq_pred))
median(rank_results)
