#!/usr/bin/python3
import pyspark.sql.functions as f

from math import sqrt

"""custom modules"""
import utils
import schemas


def calc_F(df_clust, G=1):
    """calculate the force per unit mass (F) acting on every particle"""

    df_clust_cartesian = utils.df_x_cartesian(df_clust, filterCol="id")


    df_F_cartesian = df_clust_cartesian.selectExpr("id", "id_other", "m_other",
                                                   "`x` - `x_other` as `diff(x)`",
                                                   "`y` - `y_other` as `diff(y)`",
                                                   "`z` - `z_other` as `diff(z)`"
                                                   )
    df_F_cartesian = df_F_cartesian.selectExpr("id", "id_other",
                                               "`diff(x)` * `m_other` as `num(x)`",
                                               "`diff(y)` * `m_other` as `num(y)`",
                                               "`diff(z)` * `m_other` as `num(z)`",
                                               "abs(`diff(x)` * `diff(x)` * `diff(x)`) as `denom(x)`",
                                               "abs(`diff(y)` * `diff(y)` * `diff(y)`) as `denom(y)`",
                                               "abs(`diff(z)` * `diff(z)` * `diff(z)`) as `denom(z)`",
                                               )

    df_F_cartesian = df_F_cartesian.selectExpr("id", "id_other",
                                               "`num(x)` / `denom(x)` as `Fx`",
                                               "`num(y)` / `denom(y)` as `Fy`",
                                               "`num(z)` / `denom(z)` as `Fz`",
                                               )

    df_F = utils.df_agg_sum(df_F_cartesian, "id", "Fx", "Fy", "Fz")
    df_F = df_F.selectExpr("id", f"`Fx` * {-G} as Fx", f"`Fy` * {-G} as Fy", f"`Fz` * {-G} as Fz")

    return df_F


def calc_gforce_cartesian(df_clust, G=1):
    """calculate the distance and gravity force between every two particles in the cluster"""

    df_clust_cartesian = utils.df_x_cartesian(df_clust, filterCol="id")

    @f.udf(schemas.dist_gforce)
    def get_gravity_split(x1, x2, y1, y2, z1, z2, m1, m2):
        """
        calcualte gravity and distance force between two points in 3d space
        """
        # we've removed duplicate id-s already
        # if x1 == x2 and y1 == y2 and z1 == z2:
        #   return (0, 0, 0, 0)
        vx, vy, vz = x2 - x1, y2 - y1, z2 - z1
        dist = sqrt(vx * vx + vy * vy + vz * vz)
        gforce = (G * m1 * m2) / (dist * dist)
        return (dist, gforce, (vx / dist) * gforce, (vy / dist) * gforce, (vz / dist) * gforce)

    df_gforce_cartesian = (df_clust_cartesian
                           # https://stackoverflow.com/a/51908455/1002899
                           .withColumn("gforce", f.explode(f.array(
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
