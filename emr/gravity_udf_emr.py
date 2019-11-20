#!/usr/bin/python3
import numpy as np

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession  
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import pyspark.sql.functions as f

from math import sqrt
from scipy.constants import G

import os


"""arguments"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--limit", help="number of input rows to read", nargs='?', const=1000, type=int)
parser.add_argument("--outputDir", help="output path", nargs='?', default="../output/")
parser.add_argument("--inputDir", help="input path", nargs='?', default="../data/")
args = parser.parse_args()
"""/arguments"""

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

def load_cluster_data(fname, pth=None, header="true", limit=None, part=None):
    """load a cluster from the dataset
    https://www.kaggle.com/mariopasquato/star-cluster-simulations 
    """
    schm = StructType([StructField('x', DoubleType(), True),
                        StructField('y', DoubleType(), True),
                        StructField('z', DoubleType(), True),
                        StructField('vx', DoubleType(), True),
                        StructField('vy', DoubleType(), True),
                        StructField('vz', DoubleType(), True),
                        StructField('m', DoubleType(), True),
                        StructField('id', IntegerType(), True)])

    if pth:
        floc = os.path.join(pth, fname)
    else: #default data location
        floc = os.path.join(os.path.dirname(__file__), '..', 'data', fname)

    df = spark.read.load(floc, 
                                format="csv", header=header, schema=schm)

    if limit:
        df = df.limit(limit)
    if part:
        return df.repartition(part)
    return df


clust = load_cluster_data("c_0000.csv", pth=args.inputDir, limit=args.limit, part="id")

rdd_idLocMass = clust.select('id', 'x', 'y', 'z', 'm').rdd


allLocMass = rdd_idLocMass.collect()
rdd_idLocMass_cartesian = rdd_idLocMass.flatMap(
    lambda x: [list(x) + list(y)[1:] for y in allLocMass]
    )


df_idLocMass_cartesian = rdd_idLocMass_cartesian.toDF(['id', 'x', 'y', 'z', 'm', 'x_other', 'y_other', 'z_other', 'm_other'])


schm_g_split = StructType([
    StructField("gforce", DoubleType(), False),
    StructField("gx", DoubleType(), False),
    StructField("gy", DoubleType(), False),
    StructField("gz", DoubleType(), False)])

@udf(schm_g_split)
def get_gravity_split(x1,x2,y1,y2,z1,z2,m1,m2):
    """
    calcualte gravity force between two points in 3d space
    """
    if x1 == x2 and y1 == y2 and z1 == z2:
        return (0, 0, 0, 0)
    vx, vy, vz = x2 - x1, y2 - y1, z2 - z1
    dist = sqrt(vx*vx + vy*vy + vz*vz)
    gforce = (G*m1*m2)/(dist*dist)
    return (gforce, (vx/dist) * gforce, (vy/dist) * gforce, (vz/dist) * gforce)

df_gforce_cartesian = (df_idLocMass_cartesian
    .withColumn("gforce", 
        #https://stackoverflow.com/a/51908455/1002899
        f.explode(f.array(
            get_gravity_split(df_idLocMass_cartesian['x'],
                            df_idLocMass_cartesian['x_other'],
                            df_idLocMass_cartesian['y'],
                            df_idLocMass_cartesian['y_other'],
                            df_idLocMass_cartesian['z'],
                            df_idLocMass_cartesian['z_other'],
                            df_idLocMass_cartesian['m'],
                            df_idLocMass_cartesian['m_other']
            )
        ))
    )
    .select("id", "gforce.*")
    )


df_gforce = df_gforce_cartesian.groupBy("id").sum("gforce", "gx", "gy", "gz")\
                .withColumnRenamed("sum(gforce)","gforce").withColumnRenamed("sum(gx)","gx").withColumnRenamed("sum(gy)","gy").withColumnRenamed("sum(gz)","gz")


df_gforce.write.csv(os.path.join(args.outputDir, sc.applicationId))