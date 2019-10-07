#!/usr/bin/python3
import numpy as np

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession  
from pyspark.sql.types import *

from operator import add

import utils

"""arguments"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--limit", help="limit", nargs='?', const=1, type=int)
args = parser.parse_args()
"""/arguments"""

sc = SparkContext('local[*]')
spark = SparkSession(sc)

utl = utils.SparkUtils(spark)
clust = utl.load_cluster_data("c_0000.csv", limit=args.limit) #whole dataset is slow

rdd_idLocMass = clust.select('id', 'x', 'y', 'z', 'm').rdd

rdd_idLocMass_map = rdd_idLocMass.map(
	lambda x: (x['id'], (np.array([x['x'], x['y'], x['z']]), x['m']))
	)
allLocMass = sc.broadcast(rdd_idLocMass_map.collect())
rdd_idLocMass_map_cartesian = rdd_idLocMass_map.flatMapValues(
	lambda x: [(x, y[1]) for y in allLocMass.value]
	)
rdd_gforce_map = rdd_idLocMass_map_cartesian.mapValues(
	lambda x: utils.get_gforce(x[0][0], x[1][0], x[0][1], x[1][1])
	)

rdd_gforce = rdd_gforce_map.reduceByKey(add)

print(rdd_gforce.take(10))