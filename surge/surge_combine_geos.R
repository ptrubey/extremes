rm(list = ls())
# libs = c('sf','modeest')
libs = c('sf')
sapply(libs, require, character.only = TRUE)

tiger_path = paste0(
  Sys.getenv('USERPROFILE'), 
  '/Nextcloud/Research/CommonData/Tiger'
  )

loadsf = function(args){
  path = paste0(tiger_path, args[1])
  layer = args[2]
  return(read_sf(dsn = path, layer = layer))
}

blocks = list(
  c('/TABBLOCK/tl_2023_24_tabblock20', 'tl_2023_24_tabblock20'),
  c('/TABBLOCK/tl_2023_34_tabblock20', 'tl_2023_34_tabblock20'),
  c('/TABBLOCK/tl_2023_42_tabblock20', 'tl_2023_42_tabblock20')
  )
blocks_sfs = lapply(blocks, loadsf)
blocks_sf = do.call(rbind, blocks_sfs)
blocks_sf$housing_density = blocks_sf$HOUSING20 / blocks_sf$ALAND20
blocks_sf$housing_density[is.na(blocks_sf$housing_density)] = 0
blocks_sf$population_density = blocks_sf$POP20 / blocks_sf$ALAND20
blocks_sf$population_density[is.na(blocks_sf$population_density)] = 0
blocks_sf = blocks_sf[!(blocks_sf$ALAND20 == 0),]
blocks_sf_houdens = blocks_sf[blocks_sf$housing_density > 0,]
blocks_sf_popdens = blocks_sf[blocks_sf$population_density > 0,]

inv_quantile = function(x, q){return(ecdf(x)(quantile(x, q)))}
# houdens_break = exp(mlv(log(blocks_sf_houdens$housing_density), method = 'shorth'))
# houdens_break = exp(-6.5)
houdens_break = exp(quantile(log(blocks_sf_houdens$housing_density), 0.95))

# popdens_break = exp(mlv(log(blocks_sf_popdens$population_density), method = 'shorth'))
# popdens_break = exp(-5.2)
popdens_break = exp(quantile(log(blocks_sf_popdens$population_density), 0.95))

blocks_houdens_subset = blocks_sf_houdens[blocks_sf_houdens$housing_density > houdens_break,]
blocks_popdens_subset = blocks_sf_popdens[blocks_sf_popdens$population_density > popdens_break,]

st_write(
  blocks_sf, 
  dsn = paste0(tiger_path, '/TABBLOCK/tl_2023_NJ_MD_PA_tabblock20'), 
  layer = 'blocks',
  driver = "ESRI Shapefile",
  append = FALSE
  )
st_write(
  blocks_houdens_subset, 
  dsn = paste0(tiger_path, '/TABBLOCK/tl_2023_NJ_MD_PA_tabblock20'), 
  layer = 'blocks_hd',
  driver = "ESRI Shapefile",
  append = FALSE
  )
st_write(
  blocks_popdens_subset, 
  dsn = paste0(tiger_path, '/TABBLOCK/tl_2023_NJ_MD_PA_tabblock20'), 
  layer = 'blocks_pd',
  driver = "ESRI Shapefile",
  append = FALSE
  )

landmarks = list(
  c('/POINTLM/tl_2023_24_pointlm', 'tl_2023_24_pointlm'),
  c('/POINTLM/tl_2023_34_pointlm', 'tl_2023_34_pointlm'),
  c('/POINTLM/tl_2023_42_pointlm', 'tl_2023_42_pointlm')
  )
landmarks_sfs = lapply(landmarks, loadsf)
landmarks_sf = do.call(rbind, landmarks_sfs)
# Removing mountain tops, cul-de-sacs, etc.
landmarks_sf_subset = landmarks_sf[
  !(landmarks_sf$MTFCC %in% c('C3022','C3061','C3062','C3066','C3071','')),
  ]

st_write(
  landmarks_sf_subset,
  dsn = paste0(tiger_path, '/POINTLM/tl_2023_NJ_MD_PA_pointlm'),
  layer = 'landmarks',
  driver = 'ESRI Shapefile',
  append = FALSE
  )