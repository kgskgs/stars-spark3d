#!/usr/bin/python3
import pyspark.sql.functions as F
from pyspark.sql import Window

import utils


def calc_cm(df_clust):
    """Calcuate the center of mass of the cluster
    in our dataset all the masses are equal, so
    it is equal to the mean of the coordinates::

                 N
        R = 1/M *Σ m_i*r_i
                i=1

    :param df_clust: cluster data - position, velocity, and mass
    :type df_clust: pyspark.sql.DataFrame, with schema schemas.clust_input
    :returns: center of mass
    :rtype: pyspark.sql.types.Row
    """

    df_cm = df_clust.selectExpr("m", "`x` * `m` as `xm`",
                                "`y` * `m` as `ym`",
                                "`z` * `m` as `zm`",)
    # sum
    df_cm = df_cm.groupBy().sum()
    df_cm = df_cm.selectExpr("`sum(xm)` / `sum(m)` as `x`",
                             "`sum(ym)` / `sum(m)` as `y`",
                             "`sum(zm)` / `sum(m)` as `z`",)
    return df_cm.collect()[0]


def calc_rh(df_clust, cm):
    """Calculate the half-mass radius of the cluster

    :param df_clust: cluster data - position, velocity, and mass
    :type df_clust: pyspark.sql.DataFrame, with schema schemas.clust_input
    :param cm: cluster center of mass
    :type cm: iterable ([x, y, z])
    :returns: half-mass radius
    :rtype: float
    """

    df_dist = df_clust.selectExpr("id", "m",
                                  f"{cm[0]} - `x` as dx",
                                  f"{cm[1]} - `y` as dy",
                                  f"{cm[2]} - `z` as dz",
                                  )

    df_dist = df_dist.selectExpr("id", "m",
                                 "sqrt(`dx` * `dx` + `dy` * `dy` + `dz` * `dz`) as dist_c"
                                 )

    # https://stackoverflow.com/a/50144436/1002899
    w = Window.orderBy("dist_c").rowsBetween(
        Window.unboundedPreceding,  # Take all rows from the beginning of frame
        Window.currentRow           # To current row
    )

    df_m_cumulative = df_dist.withColumn('total_m_from_center', F.sum("m").over(w)).repartition("id")

    M = df_m_cumulative.selectExpr("max(`total_m_from_center`) as `m`").collect()[0][0]
    df_m_cumulative = df_m_cumulative.where(f"`total_m_from_center` >= {M/2}")
    rh = df_m_cumulative.orderBy("dist_c").limit(1).collect()[0]["dist_c"]

    return rh


def calc_T(df_clust, G=1):
    """Calculate the total kinetic energy of the cluster::

                N
        T = 1/2*Σ m_i*v_i^2
               i=1

    [Aarseth, S. (2003). Gravitational N-Body Simulations: Tools and Algorithms
    eq. (1.2)]

    :param df_clust: cluster data - position, velocity, and mass
    :type df_clust: pyspark.sql.DataFrame, with schema schemas.clust_input
    :param G: gravitational constant to use, defaults to 1
    :type G: float, optional
    :returns: T
    :rtype: float
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
    """Calculate the total potential energy of the cluster::

          N   N   G*m_i*m_j
        - Σ   Σ  -----------
         i=1 j>i |r_i - r_j|

    [Aarseth, S. (2003). Gravitational N-Body Simulations: Tools and Algorithms
    eq. (1.2)]

    :param df_clust: cluster data - position, velocity, and mass
    :type df_clust: pyspark.sql.DataFrame, with schema schemas.clust_input
    :param G: gravitational constant to use, defaults to 1
    :type G: float, optional
    :returns: U
    :rtype: float
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
    """Calculate the total energy of the cluster
    in our case there is no external energy (W)

    [Aarseth, S. (2003). Gravitational N-Body Simulations: Tools and Algorithms
    eq. (1.5)]

    :param df_clust: cluster data - position, velocity, and mass
    :type df_clust: pyspark.sql.DataFrame, with schema schemas.clust_input
    :param G: gravitational constant to use, defaults to 1
    :type G: float, optional
    :param W: external energy, defaults to 0
    :type W: float, optional
    :returns: E
    :rtype: float
    """

    return calc_T(df_clust) + calc_U(df_clust) + W


def calc_J(df_clust):
    """Calculate the total angular momentum of the cluster::

            N
        J = Σ r_i × m_i * v_i
           i=1

    [Aarseth, S. (2003). Gravitational N-Body Simulations: Tools and Algorithms
    eq. (1.3)]

    :param df_clust: cluster data - position, velocity, and mass
    :type df_clust: pyspark.sql.DataFrame, with schema schemas.clust_input
    :returns: J, broken into components
    :rtype: {[type]}
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
