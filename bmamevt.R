# remotes::install_github("lbelzile/BMAmevt") 
rm(list = ls())
libs = c('BMAmevt', 'coda')
sapply(libs, require, character.only = TRUE)
rm(libs)

MCpar = list(sd = 0.1)
data(pb.Hpar)

fit_pairwise_betas <- function(path, nsim, nburn, nper){
  df = read.csv(path)
  dfs = df / apply(df, 1, sum) # project to simplex
  model = posteriorMCMC(
    Nsim = nsim, 
    Nbin = nburn,
    dat = dfs,
    prior = prior.pb,
    proposal = proposal.pb, 
    likelihood = dpairbeta,
    Hpar = pb.Hpar,
    MCpar = MCpar
    )
  nCol = ncol(dfs); nSamp = nrow(model$stored.vals)
  f = function(x){rpairbeta(n = nper, dimData = ncol(dfs), par = x)}
  out = array(t(apply(model$stored.vals, 1, f)), dim = c(nper * nSamp, nCol))
  return(out)
}

path = '~/git/projgamma/simulated/sphere/data_m5_r5_i0.csv'
path = '~/git/projgamma/simulated/sphere/data_m2_r12_i0.csv'


heidel.diag(model$stored.vals)
ppred = rpairbeta(1, dimData = ncol(dfs), par = model$stored.vals[25001,])

# leeds
