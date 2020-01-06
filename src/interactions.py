#!/usr/bin/python3

import utils

def calc_F(df_clust, G=1):
    """calculate the force per unit mass acting on every particle

             N    m_j*(r_i - r_j)
    F = -G * Σ    ---------------
            i!=j   |r_i - r_j|^3

    r - position
    m - mass
    G - Gravitational Force Constant

    [Aarseth, S. (2003). Gravitational N-Body Simulations: Tools and Algorithms]
    """

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
    df_F = df_F.selectExpr("id",
                           f"`Fx` * {-G} as Fx",
                           f"`Fy` * {-G} as Fy",
                           f"`Fz` * {-G} as Fz")

    return df_F


def calc_gforce_cartesian(df_clust, G=1):
    """calculate the distance and gravity force between every two particles in the cluster

        G * m_1 * m_2
    F = -------------
            d^2

    m - mass
    d - distance between the centers
    G - Gravitational Force Constant

    [http://www.astronomy.ohio-state.edu/~pogge/Ast161/Unit4/gravity.html]
    """

    df_clust_cartesian = utils.df_x_cartesian(df_clust, filterCol="id")

    df_gforce_cartesian = df_clust_cartesian.selectExpr("id", "id_other",
                                                        "`x_other` - `x` as  `vx`",
                                                        "`y_other` - `y` as  `vy`",
                                                        "`z_other` - `z` as  `vz`",
                                                        f"{G} * `m` * `m_other` as `num`"
                                                        )
    df_gforce_cartesian = df_gforce_cartesian.selectExpr("id", "id_other", "vx", "vy", "vz", "num",
                                                         "sqrt(`vx` * `vx` + `vy` * `vy` + `vz` * `vz`) as `dist`"
                                                         )
    df_gforce_cartesian = df_gforce_cartesian.selectExpr("id", "id_other", "dist", "vx", "vy", "vz",
                                                         "`num` / (`dist` * `dist`) as `gforce`"
                                                         )
    df_gforce_cartesian = df_gforce_cartesian.selectExpr("id", "id_other", "dist", "gforce",
                                                         "(`vx` / `dist`) * `gforce` as `gx`",
                                                         "(`vy` / `dist`) * `gforce` as `gy`",
                                                         "(`vz` / `dist`) * `gforce` as `gz`"
                                                         )

    return df_gforce_cartesian
