#!/usr/bin/python3
from math import ceil
from pyspark.sql.functions import row_number
from pyspark.sql import Window


def calc_cm(df_clust):
    """
    calcuate the center of mass of the cluster
    in our dataset all the masses are equal, so
    it is equal to the mean of the coordinates
    """
    df_cm = df_clust.selectExpr("mean(`x`) as `x`",
                                "mean(`y`) as `y`",
                                "mean(`z`) as `z`")

    return df_cm.collect()[0]


def calc_rh(df_clust, cm):
    """
    calculate the half-mass radius of the cluster
    since all the masses are equal that is the distance
    from the center to the n/2-th closest star
    """
    half_count = ceil(df_clust.count() / 2)

    df_dist = df_clust.selectExpr("id",
                                  f"{cm[0]} - `x` as dx",
                                  f"{cm[1]} - `y` as dy",
                                  f"{cm[2]} - `z` as dz",
                                  )

    df_dist = df_dist.selectExpr("id",
                                 "sqrt(`dx` * `dx` + `dy` * `dy` + `dz` * `dz`) as dist_c"
                                 )

    df_dist = df_dist.withColumn('rown',
                                 row_number().over(Window.orderBy("dist_c")))
    return df_dist.filter(f"rown={half_count}").collect()[0]['dist_c']


def calc_T(df_clust, G=1):
    """
    calculate the total kinetic energy of the cluster

            N
    T = 1/2*Σ m_i*v_i^2
           i=1

    """

    # get magnitutde of the velocity
    df_clust = df_clust.selectExpr("m",
                                   "sqrt("
                                   "`vx` * `vx` + `vy` * `vy` + `vz` * `vz`"
                                   ") as `v`")

    df_T = df_clust.selectExpr("`m` * `v` * `v` as `T`")
    T = df_T.groupBy().sum().collect()[0][0] / 2

    return T


"""
  N   N   G*m_i*m_j
- Σ   Σ  -----------
 i=1 j>i |r_i - r_j|
"""
