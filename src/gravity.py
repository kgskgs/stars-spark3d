#!/usr/bin/python3
from pyspark.sql.functions import udf
import pyspark.sql.functions as f

import numpy as np
from math import sqrt

"""custom modules"""
import utils
import schemas


def calc_gforce_cartesian(in_name, in_path, limit, G, part="id"):
    """calculate the distance and gravity force between every two particles in the cluster"""
    df_clust = utils.load_df(in_name, in_path, limit=limit, schema=schemas.clust_input, part=part)

    df_clust_cartesian = df_clust.crossJoin(
        df_clust.selectExpr(*["`{0}` as {0}_other".format(x) for x in df_clust.columns])
        ).filter("id != id_other")

    @udf(schemas.dist_gforce)
    def get_gravity_split(x1,x2,y1,y2,z1,z2,m1,m2):
        """
        calcualte gravity and distance force between two points in 3d space
        """
        #we've removed duplicate id-s already
        #if x1 == x2 and y1 == y2 and z1 == z2:
        #   return (0, 0, 0, 0)
        vx, vy, vz = x2 - x1, y2 - y1, z2 - z1
        dist = sqrt(vx*vx + vy*vy + vz*vz)
        gforce = (G*m1*m2)/(dist*dist)
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

    return df_gforce_cartesian

def sum_gforce(in_name, in_path=None, part="id"):
    """sum the cartesian data by id to get the effective force acting on one particle"""
    if in_path:
        df_gforce_cartesian = utils.load_df(in_name, in_path, schema=schemas.dist_gforce_cartesian, part=part)
    else:
        df_gforce_cartesian = in_name

    df_gforce = df_gforce_cartesian.groupBy("id").sum("gforce", "gx", "gy", "gz")\
            .selectExpr("id","`sum(gforce)` as gforce","`sum(gx)` as gx","`sum(gy)` as gy","`sum(gz)` as gz")

    return df_gforce

