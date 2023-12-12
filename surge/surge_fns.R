rm(list = ls())
libs = c('sf', 'raster')
sapply(libs, require, character.only = TRUE)

# Function Definitions
subset_geo_to_data_boundary = function(points, shapes, ...){
  # Subsets shapefile to the data boundary (+ 100 meters or so)
  #   Intersections go faster with less geometries to deal with.
  #   Assumes points is st_multipoint class or matrix/df with names x,y; 
  #   shapes is st_polygon class
  shp_subset = st_intersection(shapes, find_data_boundary(points, ...))
  return(shp_subset)
}

find_data_boundary = function(data, epsilon = 0.001){
  # Finds the data boundary (+ 100 meters or so)
  #   Returns a polygon of the bounding box
  #   Assumes coords_point is st_multipoint class, or data.frame with
  #   names x,y for long, lat
  if (class(data)[1] == 'sf'){
    ext = st_bbox(coords_point)
  } else if(all(c('x','y') %in% names(as.data.frame(data)))){
    data = as.data.frame(data)
    ext = st_bbox(
      c(xmin = min(data$x) - epsilon,
        xmax = max(data$x) + epsilon,
        ymin = min(data$y) - epsilon,
        ymax = max(data$y) + epsilon),
      crs = 4269 # NAD83
      )
  }
  return(st_as_sfc(ext))
}

intersect_points_with_geo = function(points, shapes){
  # Intersects points with geographies.
  #   points as matrix/data frame w/ names x,y for long, lat
  #   shapes as sf
  points_sf = st_as_sf(as.data.frame(points), coords = c('x','y'), crs = 4269)
  subset_geo = subset_geo_to_data_boundary(points, shapes)
  intersects = st_intersects(points_sf, subset_geo)
  points_sf_sub = points_sf[sapply(intersects, any),]
  intersected = st_intersection(points_sf_sub, subset_geo)
  # Randomly keep only 1 obs per geography
  shuffled = intersected[sample(1:nrow(intersected), nrow(intersected)),]
  subsetted = shuffled[!duplicated(shuffled$GEOIDFQ),]
  subset_ordered = subsetted[order(subsetted$ID),]
  return(subset_ordered)
}

subset_data_points_geo = function(points, shapes, data){
  # Subsets the points by intersecting with geography; 
  #   keeps only 1 point per geography;
  #   Identifies which columns on data to keep
  #   Returns list(points = subset_points, data = subset_data)
  points = as.data.frame(points)
  points$ID = 1:nrow(points)
  
  subset_points = intersect_points_with_geo(points, shapes)
  subset_data   = data[,subset_points$ID]
  
  return(list(points = subset_points, data = subset_data))
}

voronoi_buffer = function(pts, points, buffer_size = 70){
  # Assume points has a bounding box on it.
  # Buffer size in meters
  voro = st_voronoi(pts, envelope = find_data_boundary(points))
  buffer = st_buffer(pts, buffer_size)
  intersected = st_intersection(voro, buffer)
  vb = intersected[intersected$POINTID == intersected$POINTID.1,]  
  return(vb)
}

intersect_points_with_landmarks = function(points, landmarks){
  points_sf = st_as_sf(as.data.frame(points), coords = c('x','y'), crs = 4269)
  subset_lm = subset_geo_to_data_boundary(points, landmarks)
  buffered = voronoi_buffer(subset_lm, points) # points to pass along bounding box
  intersects = st_intersects(points_sf, buffered)
  points_sf_sub = points_sf[sapply(intersects, any),]
  intersected = st_intersection(points_sf_sub, buffered)
  shuffled = intersected[sample(1:nrow(intersected), nrow(intersected)),]
  subsetted = shuffled[!duplicated(shuffled$POINTID),]
  subset_ordered = subsetted[order(subsetted$ID),]
  return(subset_ordered)
}

subset_data_points_landmarks = function(points, landmarks, data){
  # Subsets the points by intersecting with geography; 
  #   keeps only 1 point per geography;
  #   Identifies which columns on data to keep
  #   Returns list(points = subset_points, data = subset_data)
  points = as.data.frame(points)
  points$ID = 1:nrow(points)
  
  subset_points = intersect_points_with_landmarks(points, landmarks)
  subset_data   = data[,subset_points$ID]
  
  return(list(points = subset_points, data = subset_data))
}

# EOF