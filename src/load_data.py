#!/usr/bin/python3
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *

from pyspark3d.repartitioning import prePartition
from pyspark3d.repartitioning import repartitionByCol
from pyspark3d.visualisation import scatter3d_mpl

import pylab as pl
import numpy as np

sc = SparkContext('local')
spark = SparkSession(sc)


schm = StructType([StructField('x', DoubleType(), True),
                   StructField('y', DoubleType(), True),
                   StructField('z', DoubleType(), True),
                   StructField('vx', DoubleType(), True),
                   StructField('vy', DoubleType(), True),
                   StructField('vz', DoubleType(), True),
                   StructField('m', DoubleType(), True),
                   StructField('id', IntegerType(), True)])


df = spark.read.load("../data/c_0000.csv", 
                    format="csv", header="true", schema=schm)

#test spark3d module
df.show(5)

options = {
    "geometry": "points",
    "colnames": "x,y,z",
    "coordSys": "cartesian",
    "gridtype": "onion"}

df_colid = prePartition(df, options, numPartitions=20)

df_colid.show(5)

df_repart = repartitionByCol(df_colid, "partition_id", preLabeled=True, numPartitions=20)

df_repart.show(5)

#view data
title="cluster_0000"
fnout="cluster_0000.png"

coords = np.transpose(df.select("x", "y", "z").collect())

scatter3d_mpl(coords[0], coords[1], coords[2],label="Data set", **{"facecolors":"blue", "marker": "."})

pl.title(title)
#pl.savefig(fnout)
pl.show()