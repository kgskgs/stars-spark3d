#!/usr/bin/python3
import numpy as np

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession  
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import pyspark.sql.functions as f

from math import sqrt
from scipy.constants import G as scipy_G

import os

"""custom modules"""
import utils


"""arguments"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--limit", help="number of input rows to read", nargs='?', const=1000, type=int)
parser.add_argument("--outputDir", help="output path", nargs='?', default="../output/")
parser.add_argument("--inputDir", help="input path", nargs='?', default="../data/")
parser.add_argument("-G", help="gravitational constant for the simulation", nargs='?', default=scipy_G , type=float)
args = parser.parse_args()
"""/arguments"""

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)


schm = StructType([StructField('x', DoubleType(), False),
                    StructField('y', DoubleType(), False),
                    StructField('z', DoubleType(), False),
                    StructField('vx', DoubleType(), False),
                    StructField('vy', DoubleType(), False),
                    StructField('vz', DoubleType(), False),
                    StructField('m', DoubleType(), False),
                    StructField('id', IntegerType(), False)])

df_clust = utils.load_df("c_0000.csv", pth=args.inputDir, limit=args.limit, schema=schm, part="id")

df_clust_cartesian = df_clust.crossJoin(df_clust.selectExpr(*["{0} as {0}_other".format(x) for x in df_clust.columns]))
df_clust_cartesian = df_clust_cartesian.filter("id != id_other")

schm_g_split = StructType([
    StructField("dist", DoubleType(), False),
    StructField("gforce", DoubleType(), False),
    StructField("gx", DoubleType(), False),
    StructField("gy", DoubleType(), False),
    StructField("gz", DoubleType(), False)])

@udf(schm_g_split)
def get_gravity_split(x1,x2,y1,y2,z1,z2,m1,m2):
    """
    calcualte gravity force between two points in 3d space
    """
    #we've removed duplicate id-s already
    #if x1 == x2 and y1 == y2 and z1 == z2:
    #   return (0, 0, 0, 0)
    vx, vy, vz = x2 - x1, y2 - y1, z2 - z1
    dist = sqrt(vx*vx + vy*vy + vz*vz)
    gforce = (args.G*m1*m2)/(dist*dist)
    return (dist, gforce, (vx/dist) * gforce, (vy/dist) * gforce, (vz/dist) * gforce)

df_gforce_cartesian = (df_clust_cartesian
    .withColumn("gforce", 
        #https://stackoverflow.com/a/51908455/1002899
        f.explode(f.array(
            get_gravity_split(df_clust_cartesian['x'],
                            df_clust_cartesian['x_other'],
                            df_clust_cartesian['y'],
                            df_clust_cartesian['y_other'],
                            df_clust_cartesian['z'],
                            df_clust_cartesian['z_other'],
                            df_clust_cartesian['m'],
                            df_clust_cartesian['m_other']
            )
        ))
    )
    .select("id", "id_other", "gforce.*")
    )


#get rid of reciprocal relationships when saving
utils.save_df(df_gforce_cartesian.filter("id < id_other"), "gforce_cartesian", args.outputDir)


df_gforce = df_gforce_cartesian.groupBy("id").sum("gforce", "gx", "gy", "gz")\
                .withColumnRenamed("sum(gforce)","gforce").withColumnRenamed("sum(gx)","gx").withColumnRenamed("sum(gy)","gy").withColumnRenamed("sum(gz)","gz")

utils.save_df(df_gforce, "gforce_sum", args.outputDir)