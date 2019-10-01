#!/usr/bin/python3
import numpy as np
from scipy.constants import G

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession  
from pyspark.sql.types import *



import utils

def get_gforce(coords1, coords2, mass1, mass2):
    """calculate gravitational force between two points"""
    vec = utils.ptv(coords1, coords2)
    dist = utils.vlen3d(vec)
    return utils.normv(vec) * G*mass1*mass2/dist**2

def accumulate_gforce(row):
    index = row[0]
    if index%500 == 0: print(index)
    coords, mass = row[1]
    n_rows = len(allLocMass.value)
    tmp_array = np.zeros((n_rows,3))

    for other in range(index+1, n_rows):
        otherCoords, otherMass = allLocMass.value[other]
        gforce_vector = get_gforce(coords, otherCoords, mass, otherMass)

        tmp_array[index] += gforce_vector
        tmp_array[other] -= gforce_vector

    accGforce.add(tmp_array)

sc = SparkContext('local')
spark = SparkSession(sc)

utl = utils.SparkUtils(spark)

clust = utl.load_cluster_data("c_0000.csv")
clust_index = utils.df_add_index(clust, 'id')

accGforce = sc.accumulator(np.zeros((clust.count(), 3)), utils.NpAccumulatorParam())

rdd_indLocMass = clust_index.select('index','x', 'y', 'z', 'm').rdd
rdd_indLocMass_map = rdd_indLocMass.map(
    lambda x: (x['index'], (np.array([x['x'], x['y'], x['z']]),x['m']))
    )
#make sure order will match the order of the rdd
tmpList = rdd_indLocMass_map.sortByKey().collect()

allLocMass = sc.broadcast([x[1] for x in tmpList])

#rdd_indLocMass_map.foreach(accumulate_gforce)


