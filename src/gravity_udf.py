import numpy as np

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession  
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import pyspark.sql.functions as f

from math import sqrt
from scipy.constants import G

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

clust = utl.load_cluster_data("c_0000.csv", limit=args.limit)

rdd_idLocMass = clust.select('id', 'x', 'y', 'z', 'm').rdd

allLocMass = rdd_idLocMass.collect()
rdd_idLocMass_cartesian = rdd_idLocMass.flatMap(
    lambda x: [list(x) + list(y)[1:] for y in allLocMass]
    )

df_idLocMass_cartesian = rdd_idLocMass_cartesian.toDF(['id', 'x', 'y', 'z', 'm', 'x_other', 'y_other', 'z_other', 'm_other'])


schm_g_split = StructType([
    StructField("gx", DoubleType(), False),
    StructField("gy", DoubleType(), False),
    StructField("gz", DoubleType(), False)])

@udf(schm_g_split)
def get_gravity_split(x1,x2,y1,y2,z1,z2,m1,m2):
    if x1 == x2 and y1 == y2 and z1 == z2:
        return (0, 0, 0)
    vx, vy, vz = x2 - x1, y2 - y1, z2 - z1
    dist = sqrt(vx*vx + vy*vy + vz*vz)
    gforce = (G*m1*m2)/(dist*dist)
    return ((vx/dist) * gforce, (vy/dist) * gforce, (vz/dist) * gforce)

df_gforce_cartesian = (df_idLocMass_cartesian
    .withColumn("gforce", 
        #https://stackoverflow.com/a/51908455/1002899
        f.explode(f.array(
            get_gravity_split(
            df_idLocMass_cartesian['x'],df_idLocMass_cartesian['x_other'],df_idLocMass_cartesian['y'],df_idLocMass_cartesian['y_other'],df_idLocMass_cartesian['z'],df_idLocMass_cartesian['z_other'],df_idLocMass_cartesian['m'],df_idLocMass_cartesian['m_other']
            )
        ))
    )
    .select("id", "gforce.*")
    )

df_gforce = df_gforce_cartesian.groupBy("id").sum("gx", "gy", "gz")

df_gforce.show()