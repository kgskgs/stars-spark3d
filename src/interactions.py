#!/usr/bin/python3

import utils


def calc_F(df_clust, G=1):
    """
    calculate the force per unit mass acting on every particle

             N    m_j*(r_i - r_j)
    F = -G * Σ    ---------------
            i!=j   |r_i - r_j|^3

    r - position
    m - mass
    G - Gravitational Force Constant

    [Aarseth, S. (2003). Gravitational N-Body Simulations: Tools and Algorithms
    eq. (1.1)]
    """

    df_F_cartesian = calc_F_cartesian(df_clust)

    df_F = utils.df_agg_sum(df_F_cartesian, "id", "Fx", "Fy", "Fz")
    df_F = df_F.selectExpr("id",
                           f"`Fx` * {-G} as Fx",
                           f"`Fy` * {-G} as Fy",
                           f"`Fz` * {-G} as Fz")

    return df_F


def calc_F_cartesian(df_clust):
    """
    The pairwise calculations to be used for calculating F
    can be used to check which particle(s) contribute the most
    to the effective force acting on a single one
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

    return df_F_cartesian


def step_v(df_clust, df_F, dt):
    """
    calculate v for a single timestep t, dt = t - t_0
    v_i(t) = F_i*∆t + v_i(t0)

    [Aarseth, S. (2003). Gravitational N-Body Simulations: Tools and Algorithms
    eq. (1.19)]
    """
    df_F = df_F.selectExpr("id",
                           f"`Fx`*{dt} as `vx`",
                           f"`Fy`*{dt} as `vy`",
                           f"`Fz`*{dt} as `vz`"
                           )

    df_v_t = utils.df_elementwise(df_F, df_clust, "id", "+",
                                  "vx", "vy", "vz")

    return df_v_t


def step_r(df_clust, df_F, dt):
    """
    calculate r for a single timestep t, dt = t - t_0
    r_i(t) = 1/2*F_i*∆t^2 + v_i(t0)*∆t + r_i(t0)

    [Aarseth, S. (2003). Gravitational N-Body Simulations: Tools and Algorithms
    eq. (1.19)]
    """

    df_F = df_F.selectExpr("id",
                           f"`Fx`*{dt}*{dt}/2 as `x`",
                           f"`Fy`*{dt}*{dt}/2 as `y`",
                           f"`Fz`*{dt}*{dt}/2 as `z`"
                           )

    df_v0 = df_clust.selectExpr("id",
                                f"`vx` * {dt} as `x`",
                                f"`vy` * {dt} as `y`",
                                f"`vz` * {dt} as `z`"
                                )

    df_r_t = utils.df_elementwise(df_clust, df_v0, "id", "+",
                                  "x", "y", "z")

    df_r_t = utils.df_elementwise(df_r_t, df_F, "id", "+",
                                  "x", "y", "z")

    return df_r_t


def calc_gforce_cartesian(df_clust, G=1):
    """
    calculate the distance and gravity force between every two particles in the cluster

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
