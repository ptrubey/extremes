rm(list = ls())
source('./surge_fns.R')

# Getting TIGER data
tiger_path = paste0(Sys.getenv('USERPROFILE'), '/Nextcloud/Research/CommonData/Tiger')
blocks_hd = read_sf(
  dsn = paste0(tiger_path, '/TABBLOCK/tl_2023_NJ_MD_PA_tabblock20'),
  layer = 'blocks_hd'
)
blocks_pd = read_sf(
  dsn = paste0(tiger_path, '/TABBLOCK/tl_2023_NJ_MD_PA_tabblock20'),
  layer = 'blocks_pd'
)

# Loading SLOSH data
load('./datasets/slosh_dat_nj.rda')
# coords in grid of approximately 86.5 meters X, 110.6 meters Y
res_hd = subset_data_points_geo(coords, blocks_hd, out)
res_pd = subset_data_points_geo(coords, blocks_pd, out)


