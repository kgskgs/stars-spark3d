#!/usr/bin/python3
import numpy as np


from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession  
from pyspark.sql.types import *

import utils

"""arguments"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--limit", help="limit", nargs='?', const=1, type=int)
args = parser.parse_args()
"""/arguments"""

def accumulate_gforce(row):
    #row format - (id, (coordinates, mass))
    index = row[0]
    coords, mass = row[1]
    n_rows = len(allLocMass.value)
    
    tmp_array = np.zeros((n_rows,3))

    for other in range(index+1, n_rows):
        otherCoords, otherMass = allLocMass.value[other]
        gforce_vector = utils.get_gforce(coords, otherCoords, mass, otherMass)

        tmp_array[index] += gforce_vector
        tmp_array[other] -= gforce_vector

    accGforce.add(tmp_array)

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

utl = utils.SparkUtils(spark)
clust = utl.load_cluster_data("c_0000.csv", limit=args.limit) #whole dataset is slow

#WARN WindowExec: No Partition Defined for Window operation! 
#Moving all data to a single partition, 
#this can cause serious performance degradation.
clust_index = utils.df_add_index(clust, 'id')

accGforce = sc.accumulator(np.zeros((clust.count(), 3)), utils.NpAccumulatorParam())

rdd_indLocMass = clust_index.select('index','x', 'y', 'z', 'm').rdd
rdd_indLocMass_map = rdd_indLocMass.map(
    lambda x: (x['index'], (np.array([x['x'], x['y'], x['z']]),x['m']))
    )
#make sure order will match the order of the rdd
rdd_indLocMass_map = rdd_indLocMass_map.sortByKey()
tmpList = rdd_indLocMass_map.collect()

allLocMass = sc.broadcast([x[1] for x in tmpList])

#this forces evaltion - this method can't be lazy
rdd_indLocMass_map.foreach(accumulate_gforce)

#print(accGforce.value)
