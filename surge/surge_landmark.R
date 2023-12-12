rm(list = ls())
source('./surge_fns.R')

# Getting TIGER data
tiger_path = paste0(Sys.getenv('USERPROFILE'), '/Nextcloud/Research/CommonData/Tiger')
landmarks = read_sf(
  dsn = paste0(tiger_path, '/POINTLM/tl_2023_NJ_MD_PA_pointlm'),
  layer = 'landmarks'
)

# Loading SLOSH data
load('./datasets/slosh_dat_nj.rda')

res_lm = subset_data_points_landmarks(coords, landmarks, out)
