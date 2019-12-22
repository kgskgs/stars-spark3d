#!/usr/bin/python3
import pyspark.sql.functions as f

from math import sqrt

"""custom modules"""
import utils
import schemas


def calc_f (df_clust, G=1):
    """calculate the distance and gravity force between every two particles in the cluster"""
    pass


def calc_gforce_cartesian(df_clust, G=1):
    """calculate the distance and gravity force between every two particles in the cluster"""

    df_clust_cartesian = utils.df_x_cartesian(df_clust, filterCol="id")

    @f.udf(schemas.dist_gforce)
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
