#!/usr/bin/python3
import utils


class Simulation:
    """
    set the simulation conditions

    currently available computation methods:
    *eul1 - basic step method
    *eul2 - version of eul1 where average force over the period dt is used

    :param cluster: cluster data - position and velocity [broken into componenets], and mass
    :type cluster: pyspark.sql.DataFrame, with schema schemas.clust_input
    :param integrator: Integration method to use
    :type integrator: TODO
    :param ttarget: target time to reach when running the simulation
    :type ttarget: int
    :param G: gravitational constant to use, defaults to 1
    :type G: float, optional
    :param t: timestamp of the current cluster data, defaults to 0
    :type t: int, optional
    """

    def __init__(self, cluster, integrator, ttarget, G=1, t=0):
        """Constructor"""
        self.t = t
        self.ttarget = ttarget

        self.cluster = cluster

        self.G = G

        self.integrator = integrator

    def run(self):
        """
        run the simulation with the chosen method until the target time is reached


        :raises: ValueError if the target time is already reached
        """
        if self.t >= self.ttarget:
            raise ValueError("Target time is already reached")

        while (self.t < self.ttarget):
            newSnapshot, timePassed = self.integrator.advance(self.cluster)
            self.cluster = newSnapshot
            self.cluster.localCheckpoint()
            self.t += timePassed



class Intergrator_Euler:
    """
    Bruteforce integration method that advances all particles
    at a constant time step (dt)

    :param dt: time step for the simulation
    :type dt: float
    :param nparts: number of partitions to use for dataframes that will undergo cartesian multiplication with themselves
    :type nparts: int
    :param G: gravitational constant to use, defaults to 1
    :type G: float, optional
    :param t: timestamp of the current cluster data, defaults to 0
    :type t: int, optional
    """

    def __init__(self, dt, nparts, G=1):
        """Constructor"""

        self.dt = dt

        self.G = G

        self.nparts = nparts

    def advance(self, df_clust):
        """
        Advance by a step

        :param insteps: number of steps to advance
        :type insteps: int
        :returns: the new positions and velocities of the particles of the cluster, and time passed
        :rtype: tuple (pyspark.sql.DataFrame, with schema schemas.clust_input, float)
        """

        df_clust = df_clust.localCheckpoint().repartition(self.nparts, "id")
        df_F = self.calc_F(df_clust).localCheckpoint()
        df_v, df_r = self.step_v(df_clust, df_F), self.step_r(
            df_clust, df_F)

        df_clust = df_r.join(df_v, "id").join(
            df_clust.select("id", "m"), "id")

        return (df_clust, self.dt)

    def calc_F(self, df_clust):
        """
        calculate the force per unit mass acting on every particle

                 N    m_j*(r_i - r_j)
        F = -G * Σ   -----------------
                i!=j  |r_i - r_j|^3

        r - position
        m - mass
        G - Gravitational Force Constant

        [Aarseth, S. (2003). Gravitational N-Body Simulations: Tools and Algorithms
        eq. (1.1)]

        :param df_clust: cluster data - position, velocity, and mass
        :type df_clust: pyspark.sql.DataFrame, with schema schemas.clust_input
        :returns: force per unit of mass
        :rtype: pyspark.sql.DataFrame, with schema schemas.F_id
        """

        df_F_cartesian = self.calc_F_cartesian(df_clust)

        df_F = utils.df_agg_sum(df_F_cartesian, "id", "Fx", "Fy", "Fz")
        df_F = df_F.selectExpr("id",
                               f"`Fx` * {-self.G} as Fx",
                               f"`Fy` * {-self.G} as Fy",
                               f"`Fz` * {-self.G} as Fz")

        return df_F

    def calc_F_cartesian(self, df_clust):
        """
        The pairwise calculations to be used for calculating F
        can be used to check which particle(s) contribute the most
        to the effective force acting on a single one

        :param df_clust: cluster data - position, velocity, and mass
        :type df_clust: pyspark.sql.DataFrame, with schema schemas.clust_input
        :returns: the forces acting between every two particles
        :rtype: pyspark.sql.DataFrame, with schema schemas.F_cartesian
        """
        df_clust_cartesian = utils.df_x_cartesian(
            df_clust, ffilter="id != id_other")

        df_F_cartesian = df_clust_cartesian.selectExpr("id", "id_other", "m_other",
                                                       "`x` - `x_other` as `diff(x)`",
                                                       "`y` - `y_other` as `diff(y)`",
                                                       "`z` - `z_other` as `diff(z)`"
                                                       )
        df_F_cartesian = df_F_cartesian.selectExpr("id", "id_other",
                                                   "`diff(x)` * `m_other` as `num(x)`",
                                                   "`diff(y)` * `m_other` as `num(y)`",
                                                   "`diff(z)` * `m_other` as `num(z)`",
                                                   "sqrt(`diff(x)` * `diff(x)` + `diff(y)`"
                                                   "* `diff(y)` + `diff(z)` * `diff(z)`) as `denom`",
                                                   )
        df_F_cartesian = df_F_cartesian.selectExpr("id", "id_other",
                                                   "`num(x)` / pow(`denom`, 3) as `Fx`",
                                                   "`num(y)` / pow(`denom`, 3) as `Fy`",
                                                   "`num(z)` / pow(`denom`, 3) as `Fz`",
                                                   )

        return df_F_cartesian

    def step_v(self, df_clust, df_F):
        """
        calculate v for a single timestep t, dt = t - t_0
        v_i(t) = F_i*∆t + v_i(t0)

        [Aarseth, S. (2003). Gravitational N-Body Simulations: Tools and Algorithms
        eq. (1.19)]

        :param df_clust: cluster data - position, velocity, and mass
        :type df_clust: pyspark.sql.DataFrame, with schema schemas.clust_input
        :param df_F: force per unit of mass acting on each particle
        :type df_F: pyspark.sql.DataFrame, with schema schemas.F_id
        :returns: the new velocity of each particle
        :rtype: pyspark.sql.DataFrame, with schema schemas.v_id
        """
        df_F = df_F.selectExpr("id",
                               f"`Fx`*{self.dt} as `vx`",
                               f"`Fy`*{self.dt} as `vy`",
                               f"`Fz`*{self.dt} as `vz`"
                               )

        df_v_t = utils.df_elementwise(df_F, df_clust, "id", "+",
                                      "vx", "vy", "vz")

        return df_v_t

    def step_r(self, df_clust, df_F):
        """
        calculate r for a single timestep t, dt = t - t_0
        r_i(t) = 1/2*F_i*∆t^2 + v_i(t0)*∆t + r_i(t0)

        [Aarseth, S. (2003). Gravitational N-Body Simulations: Tools and Algorithms
        eq. (1.19)]

        :param df_clust: cluster data - position, velocity, and mass
        :type df_clust: pyspark.sql.DataFrame, with schema schemas.clust_input
        :param df_F: force per unit of mass acting on each particle
        :type df_F: pyspark.sql.DataFrame, with schema schemas.F_id
        :returns: the new position of each particle
        :rtype: pyspark.sql.DataFrame, with schema schemas.r_id
        """

        df_F = df_F.selectExpr("id",
                               f"`Fx`*{self.dt}*{self.dt}/2 as `x`",
                               f"`Fy`*{self.dt}*{self.dt}/2 as `y`",
                               f"`Fz`*{self.dt}*{self.dt}/2 as `z`"
                               )

        df_v0 = df_clust.selectExpr("id",
                                    f"`vx` * {self.dt} as `x`",
                                    f"`vy` * {self.dt} as `y`",
                                    f"`vz` * {self.dt} as `z`"
                                    )

        df_r_t = utils.df_elementwise(df_clust, df_v0, "id", "+",
                                      "x", "y", "z")

        df_r_t = utils.df_elementwise(df_r_t, df_F, "id", "+",
                                      "x", "y", "z")

        return df_r_t


def calc_gforce_cartesian(df_clust, G=1):
    """
    calculate the distance and gravity force between
    every two particles in the cluster

        G * m_1 * m_2
    F = -------------
            d^2

    m - mass
    d - distance between the centers
    G - Gravitational Force Constant

    [http://www.astronomy.ohio-state.edu/~pogge/Ast161/Unit4/gravity.html]
    """

    df_clust_cartesian = utils.df_x_cartesian(
        df_clust, ffilter="id != id_other")

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
