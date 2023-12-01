import pickle
import numpy as np
import pandas as pd
import networkx as nx
import transbigdata as tbd
from sklearn.metrics.pairwise import cosine_similarity
import json


# if the distance between two amenities is less than 50m, then there is a link
def build_relation_1(sensor_amenities):
      relation = {}

      for i in range(len(sensor_amenities)):
            for j in range(len(sensor_amenities)):
                  if i != j:
                        if abs(sensor_amenities.loc[i]['geometry'].x - sensor_amenities.loc[j]['geometry'].x) < 0.0005 and \
                              abs(sensor_amenities.loc[i]['geometry'].y - sensor_amenities.loc[j]['geometry'].y) < 0.0005:
                              relation[sensor_amenities.loc[i]['amenity'], sensor_amenities.loc[j]['amenity']] = relation.get((sensor_amenities.loc[i]['amenity'], sensor_amenities.loc[j]['amenity']), 0) + 1

      return relation



# calculate distance matrix
def relation_1_to_matrix(relation):
      graph_matrix = np.zeros((len(cates), len(cates)))

      for key in relation.keys():
            graph_matrix[cates.index(key[0]), cates.index(key[1])] = relation[key]
      
      return graph_matrix



# find the nearest street for each amenity
def build_relation_2(sensor_amenities, sensor_streets):
      relation = {}
      # if sensor_amenities is not empty
      if len(sensor_amenities) != 0:
            nearest = tbd.ckdnearest_line(sensor_amenities, sensor_streets)
            for i in range(len(nearest)):
                  relation[nearest.loc[i]['amenity'], nearest.loc[i]['highway']] = relation.get((nearest.loc[i]['amenity'], nearest.loc[i]['highway']), 0) + 1

      else:
            relation = {}
            
      return relation


# calculate similarity matrix
def relation_2_to_matrix(relation):
      matrix = np.zeros((len(cates), 3))

      for key in relation.keys():
            matrix[cates.index(key[0]), ['primary', 'secondary', 'tertiary'].index(key[1])] = relation[key]
      
      # compute cosine samilarity of each line, keep 5 decimal places
      graph_matrix = np.zeros((len(cates), len(cates)))
      for i in range(len(matrix)):
            for j in range(len(matrix)):
                  graph_matrix[i][j] = round(cosine_similarity([matrix[i]], [matrix[j]])[0][0], 5)
      return graph_matrix



# map amenity to amenity_cate, for example, 'parking' to 'parking_place'
def map_amenity_to_cates(amenity):
      if amenity in Amenity_cates.keys():
            return Amenity_cates[amenity]
      


# load amenity_categories.json
json_path = 'GAN-GNN/dataset/amenity_categories.json'
with open(json_path, 'r') as f:
      Amenity_cates = json.load(f)

# get all amenity categories
cates = []
for key in Amenity_cates.keys():
      if Amenity_cates[key] not in cates:
            cates.append(Amenity_cates[key])


# load amenities file extracted from OSM
sensor_amenities_streets_path = 'GAN-GNN/dataset/amenities_dublin.pkl'
sensor_amenities_streets = pickle.load(open(sensor_amenities_streets_path, 'rb'))

# check if there is any sensor with no amenities
sensor_amenities_streets[sensor_amenities_streets['count']==0]

# remove sensor with no amenities
sensor_amenities_streets.drop(sensor_amenities_streets[sensor_amenities_streets['count']==0].index, inplace=True)

# reset index
sensor_amenities_streets.index = range(len(sensor_amenities_streets))

# create dict for each sensor
sensor_amenities_streets_dict = {}

for i in range(len(sensor_amenities_streets)):

      sensor_detail = {}
      
      sensor_detail['lat'] = sensor_amenities_streets.loc[i]['Latitude']
      sensor_detail['lon'] = sensor_amenities_streets.loc[i]['Longitude']
      sensor_detail['amenities'] = sensor_amenities_streets.loc[i]['amenities']
      sensor_detail['streets'] = sensor_amenities_streets.loc[i]['streets']
      sensor_amenities_streets_dict['Sensor'+str(i)] = sensor_detail


# data cleaning
for i in range(len(sensor_amenities_streets_dict)):
      sensor_amenities = pd.DataFrame(sensor_amenities_streets_dict['Sensor'+str(i)]['amenities'])

      # clean amenities
      #remove categories not in amenity_categories.json
      for j in range(len(sensor_amenities)):
            if sensor_amenities.loc[j]['amenity'] not in Amenity_cates.keys():
                  sensor_amenities.drop(j, inplace=True)

      # reset index
      sensor_amenities.index = range(len(sensor_amenities))

      # replace 'amenity' with 'amenity_cate'
      sensor_amenities['amenity'] = sensor_amenities['amenity'].apply(map_amenity_to_cates)

      #remove 'full_id' 'osm_type', 'amenity_lat', 'amenity_lon'
      sensor_amenities.drop(['full_id', 'osm_type', 'amenity_lat', 'amenity_lon'], axis=1, inplace=True)


      # get count of each amenity category
      count = []
      for cate in cates:
            if cate not in sensor_amenities['amenity'].values:
                  count.append(0)
            else:
                  count.append(sensor_amenities['amenity'].value_counts()[cate])


      # clean streets
      sensor_streets = pd.DataFrame(sensor_amenities_streets_dict['Sensor'+str(i)]['streets'])

      # remove 'full_id', 'osm_type'
      sensor_streets.drop(['full_id', 'osm_type'], axis=1, inplace=True)


      sensor_amenities_streets_dict['Sensor'+str(i)]['amenities'] = sensor_amenities
      sensor_amenities_streets_dict['Sensor'+str(i)]['amenities_count_list'] = count
      sensor_amenities_streets_dict['Sensor'+str(i)]['streets'] = sensor_streets



# build 2 knowledge graphs for each sensor
G_dict = {}

for i in range(len(sensor_amenities_streets_dict)):
      G_dict['Sensor'+str(i)] = {}

      G_dict['Sensor'+str(i)]['lat'] = sensor_amenities_streets_dict['Sensor'+str(i)]['lat']
      G_dict['Sensor'+str(i)]['lon'] = sensor_amenities_streets_dict['Sensor'+str(i)]['lon']
      G_dict['Sensor'+str(i)]['amenities_count_list'] = sensor_amenities_streets_dict['Sensor'+str(i)]['amenities_count_list']
      
      # build graph 1
      sensor_amenities = sensor_amenities_streets_dict['Sensor'+str(i)]['amenities']
      G_1 = nx.Graph()
      G_1.add_nodes_from(cates)
      relation_1 = build_relation_1(sensor_amenities)
      graph_matrix_1 = relation_1_to_matrix(relation_1)
      G_1.add_weighted_edges_from([(cates[i], cates[j], graph_matrix_1[i][j]) for i in range(len(cates)) for j in range(len(cates)) if graph_matrix_1[i][j] != 0])
      G_dict['Sensor'+str(i)]['G_1'] = G_1
      G_dict['Sensor'+str(i)]['graph_matrix_1'] = graph_matrix_1

      # build graph 2
      sensor_streets = sensor_amenities_streets_dict['Sensor'+str(i)]['streets']
      G_2 = nx.Graph()
      G_2.add_nodes_from(cates)
      relation_2 = build_relation_2(sensor_amenities, sensor_streets)
      graph_matrix_2 = relation_2_to_matrix(relation_2)
      G_2.add_weighted_edges_from([(cates[i], cates[j], graph_matrix_2[i][j]) for i in range(len(cates)) for j in range(len(cates)) if graph_matrix_2[i][j] != 0])
      G_dict['Sensor'+str(i)]['G_2'] = G_2
      G_dict['Sensor'+str(i)]['graph_matrix_2'] = graph_matrix_2


# save G_dict
with open('GAN-GNN/data_processing/knowledge_graphs_dublin.pkl', 'wb') as f:
      pickle.dump(G_dict, f)