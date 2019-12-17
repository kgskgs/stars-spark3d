#!/usr/bin/python3
import pyspark.sql.functions as f
from pyspark.mllib.linalg import Vector, VectorUDT
from pyspark.ml.feature import VectorAssembler

from math import sqrt

"""custom modules"""
import utils
import schemas


def calc_gforce_cartesian(in_name, in_path, limit, G, part="id"):
    """calculate the distance and gravity force between every two particles in the cluster"""
    df_clust = utils.load_df(in_name, in_path, limit=limit, schema=schemas.clust_input, part=part)

    assemblerR = VectorAssembler( inputCols=['x','y','z'], outputCol="r")
    assemblerV = VectorAssembler( inputCols=['vx','vy','vz'], outputCol="v")

    df_clust = assemblerR.transform(assemblerV.transform(df_clust)).drop('x','y','z','vx','vy','vz')

    df_clust_cartesian = df_clust.crossJoin(
        df_clust.selectExpr(*["`{0}` as {0}_other".format(x) for x in df_clust.columns])
        ).filter("id != id_other")

    @f.udf(schemas.v_dist_gforce)
    def get_gravity_split(r1,r2,m1,m2):
        """
        calcualte gravity and distance force between two points in 3d space
        """
        #we've removed duplicate id-s already
        #if x1 == x2 and y1 == y2 and z1 == z2:
        #   return (0, 0, 0, 0)
        v = r2 - r1
        dist = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
        gforce = (G*m1*m2)/(dist*dist)
        return (dist, gforce, (v/dist)*gforce)

    df_gforce_cartesian = (df_clust_cartesian
        .withColumn("gforce", 
            #https://stackoverflow.com/a/51908455/1002899
            f.explode(f.array(
                get_gravity_split(df_clust_cartesian['r'],
                                df_clust_cartesian['r_other'],
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
        df_gforce_cartesian = utils.load_df(in_name, in_path, schema=schemas.v_dist_gforce_cartesian, part=part)
    else:
        df_gforce_cartesian = in_name

    df_gforce = df_gforce_cartesian.groupBy("id").sum("gforce", "gv")\
            .selectExpr("id","`sum(gforce)` as gforce","`sum(gx)` as gx","`sum(gy)` as gy","`sum(gz)` as gz")

    return df_gforce

