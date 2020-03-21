#!/usr/bin/python3
from math import ceil
from pyspark.sql.functions import row_number
from pyspark.sql import Window

import utils


def calc_cm(df_clust):
    """
    calcuate the center of mass of the cluster
    in our dataset all the masses are equal, so
    it is equal to the mean of the coordinates

    #TODO generalize so it works with other data
             N
    R = 1/M *Σ m_i*r_i
            i=1
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

    #TODO generalize so it works with other data
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

    [Aarseth, S. (2003). Gravitational N-Body Simulations: Tools and Algorithms
    eq. (1.2)]
    """

    # get magnitutde of the velocity
    df_T = df_clust.selectExpr("m",
                               "sqrt("
                               "`vx` * `vx` + `vy` * `vy` + `vz` * `vz`"
                               ") as `v`")

    df_T = df_T.selectExpr("`m` * `v` * `v` as `T`")
    T = df_T.groupBy().sum().collect()[0][0] / 2

    return T


def calc_U(df_clust, G=1):
    """
    calculate the total potential energy of the cluster
      N   N   G*m_i*m_j
    - Σ   Σ  -----------
     i=1 j>i |r_i - r_j|

    [Aarseth, S. (2003). Gravitational N-Body Simulations: Tools and Algorithms
    eq. (1.2)]
    """

    df_clust_cartesian = utils.df_x_cartesian(df_clust,
                                              "id_other > id")

    # drop the id-s since we will squish everything anyways
    # because of the double sum
    df_U = df_clust_cartesian.selectExpr(f"{G} * `m` * `m_other` as num",
                                         "`x` - `x_other` as `diff(x)`",
                                         "`y` - `y_other` as `diff(y)`",
                                         "`z` - `z_other` as `diff(z)`")

    df_U = df_U.selectExpr("num / "
                           "sqrt(`diff(x)` * `diff(x)` + `diff(y)`"
                           "*  `diff(y)` + `diff(z)` * `diff(z)`)"
                           " as U")

    U = - df_U.groupBy().sum().collect()[0][0]

    return U


def calc_E(df_clust, G=1, W=0):
    """
    calculate the total energy of the cluster
    in our case there is no external energy (W)

    [Aarseth, S. (2003). Gravitational N-Body Simulations: Tools and Algorithms
    eq. (1.5)]
    """
    return calc_T(df_clust) + calc_U(df_clust) + W


def calc_J(df_clust):
    """
    calculate the total angular momentum of the cluster

        N
    J = Σ r_i × m_i * v_i
       i=1

    [Aarseth, S. (2003). Gravitational N-Body Simulations: Tools and Algorithms
    eq. (1.3)]
    """

    df_J = df_clust.selectExpr("x", "y", "z",
                               "`m` * `vx` as `mvx`",
                               "`m` * `vy` as `mvy`",
                               "`m` * `vz` as `mvz`"
                               )
    # cross product
    df_J = df_J.selectExpr("`y` * `mvz` - `z` * `mvy` as `Jx`",
                           "`z` * `mvx` - `x` * `mvz` as `Jy`",
                           "`x` * `mvy` - `y` * `mvx` as `Jz`")

    df_J = df_J.groupBy().sum()
    renameCols = [f"`sum({col})` as `{col}`" for col in ['Jx', 'Jy', 'Jz']]

    return df_J.selectExpr(*renameCols).collect()[0]
