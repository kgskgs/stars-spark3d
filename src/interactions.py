#!/usr/bin/python3
import utils
import cluster
import schemas
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import lit


class Simulation:
    """Set the simulation conditions


    :param cluster: cluster data - position and velocity [broken into componenets], and mass
    :type cluster: pyspark.sql.DataFrame, with schema schemas.clust_input
    :param integrator: Integration method to use
    :type integrator: TODO
    :param ttarget: target time to reach when running the simulation
    :type ttarget: int
    :param save_params: parameters to pass to utils.save_df when saving output
    :type save_params: utils.SaveOptions / dict
    :param t: timestamp of the current cluster data, defaults to 0
    :type t: int, optional
    :param dt_out: time interval between cluster snapshots, not saved if omitted
    :type dt_out: int, optional
    :param dt_diag: time interval between energy outputs, not saved if omitted
    :type dt_diag: int, optional
    """

    def __init__(self, cluster, integrator, ttarget, save_params, t=0, dt_out=None, dt_diag=None):
        """Constructor"""
        self.cluster = cluster
        self.t = t
        self.ttarget = ttarget

        self.integrator = integrator
        self.G = integrator.G

        self.dt_out = dt_out
        if dt_out:
            self.next_out = t + dt_out
        self.dt_diag = dt_diag
        if dt_diag:
            self.next_diag = t + dt_diag

        self.save_params = save_params

        self.snapshot()

    def run(self):
        """Run the simulation with the chosen method until the target time is reached

        :raises: ValueError if the target time is already reached
        """
        if self.t >= self.ttarget:
            raise ValueError("Target time is already reached")

        while (self.t < self.ttarget):
            newSnapshot, timePassed = self.integrator.advance(self.cluster)
            self.cluster = newSnapshot
            self.t += timePassed

            if self.dt_out and self.next_out <= self.t:
                self.snapshot()
                self.next_out += self.dt_out
            if self.dt_diag and self.next_diag <= self.t:
                self.diag()
                self.next_diag += self.dt_diag

    def snapshot(self, add_t=False):
        """Save a snapshot of the cluster
        """
        if add_t:
            utils.save_df(self.cluster.withColumn("t", lit(self.t)),
                          f"t{self.t}", **self.save_params)
        else:
            utils.save_df(self.cluster, f"t{self.t}", **self.save_params)

    def diag(self):
        """Save diagnostic information about the cluster energy
        """
        T, U = cluster.calc_T(self.cluster, self.G), cluster.calc_U(self.cluster, self.G)
        E = T + U
        cm = cluster.calc_cm(self.cluster)

        df_diag = SparkSession.builder.getOrCreate().createDataFrame(
            [(self.t, cm['x'], cm['y'], cm['z'], T, U, E)],
            schema=schemas.diag
        )

        utils.save_df(df_diag, f"diag_t{self.t}", **self.save_params)


class Integrator_Hermite:
    def __init__(self, dt, nparts, G=1):
        """Constructor"""

        self.dt = dt

        self.G = G

        self.nparts = nparts

        self.df_acc_jerk = None
        self.t_collision = None

    def advance(self, df_clust):

        if (not self.df_acc_jerk):
            self.df_acc_jerk = self.get_acc_jerk()

        df_clust_acc_jerk = df_clust.join(self.df_acc_jerk, "id")

        df_predict = self.predict_step(df_clust)

        df_res = self.correct_step(df_clust_acc_jerk, df_predict)

    def predict_step(self, df):

        # position
        df = df.selectExpr("id", "ax", "ay", "az", "jx", "jy", "jz",
                           f"`vx`*{self.dt} + `ax`*{self.dt}*{self.dt}/2",
                           f" + `jx`*{self.dt}*{self.dt}*{self.dt}/6 as x",

                           f"`vy`*{self.dt} + `ay`*{self.dt}*{self.dt}/2",
                           f" + `jy`*{self.dt}*{self.dt}*{self.dt}/6 as y",

                           f"`vz`*{self.dt} + `az`*{self.dt}*{self.dt}/2",
                           f" + `jz`*{self.dt}*{self.dt}*{self.dt}/6 as z")
        # velocity
        df = df.selectExpr("id", "ax", "ay", "az", "jx", "jy", "jz",
                           f"`ax`*{self.dt} + `jx`*{self.dt}*{self.dt}/2 as vx",
                           f"`ay`*{self.dt} + `jy`*{self.dt}*{self.dt}/2 as vy",
                           f"`az`*{self.dt} + `jz`*{self.dt}*{self.dt}/2 as vz")

        return df

    def correct_step(self, df_old, df_predict):

        df = utils.df_join_rename(df_predict, df_old, "id")

        # velocity
        df_v_new = df.selectExpr("id",
                                 f"`vx_other` + (ax_other + ax)*{self.dt}/2",
                                 f" + (jx_other - jx)*{self.dt}*{self.dt}/12 as vx",

                                 f"`vy_other` + (ay_other + ay)*{self.dt}/2"
                                 f" + (jy_other - jy)*{self.dt}*{self.dt}/12 as vy",

                                 f"`vz_other` + (az_other + az)*{self.dt}/2"
                                 f" + (jz_other - jz)*{self.dt}*{self.dt}/12 as vz",
                                 )

        df = df.drop("vx", "vy", "vz").join(df_v_new, "id")

        # position
        df = df.selectExpr("id", "vx", "vy", "vz",
                           "ax", "ay", "az", "jx", "jy", "jz",
                           f"`x_other` + (vx_other + vx)*{self.dt}/2"
                           f" + (ax_other - ax)*{self.dt}*{self.dt}/12",

                           f"`y_other` + (vy_other + vy)*{self.dt}/2"
                           f" + (ay_other - ay)*{self.dt}*{self.dt}/12",

                           f"`z_other` + (vz_other + vz)*{self.dt}/2"
                           f" + (az_other - az)*{self.dt}*{self.dt}/12"
                           )

    def get_acc_jerk(self, df):
        pass


class Intergrator_Euler:
    """First order integration method that advances all particles
    at a constant time step (dt). Calculates effective force per unit mass
    for each particle with O(n^2) complexity.

    Suggested nparts: # cores

    :param dt: time step for the simulation
    :type dt: float
    :param nparts: number of partitions to use for dataframes that will undergo shuffle joins
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
        """Advance by a step

        :param df_clust: cluster data - position, velocity, and mass
        :type df_clust: pyspark.sql.DataFrame, with schema schemas.clust_input
        :returns: the new positions and velocities of the particles of the cluster, and time passed
        :rtype: tuple (pyspark.sql.DataFrame, with schema schemas.clust_input, float)
        """

        df_clust = df_clust.localCheckpoint().repartition(self.nparts, "id")
        df_F = self.calc_F(df_clust).localCheckpoint()
        df_v, df_r = self.step_v(df_clust, df_F), self.step_r(
            df_clust, df_F)

        df_clust = df_r.join(df_v, "id")
        # bring order back to schema
        df_clust = df_clust.select('id', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'm')

        return (df_clust, self.dt)

    def calc_F(self, df_clust):
        """Calculate the force per unit mass acting on every particle::

                     N    m_j*(r_i - r_j)
            F = -G * Σ   -----------------
                    i!=j   |r_i - r_j|^3

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
        """The pairwise calculations to be used for calculating F
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
        """Calculate v for a single timestep t, dt = t - t_0::

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

        df_v_t = df_F.join(df_clust, "id").selectExpr(
            "id",
            f"`Fx`*{self.dt} + `vx` as `vx`",
            f"`Fy`*{self.dt} + `vy` as `vy`",
            f"`Fz`*{self.dt} + `vz` as `vz`"
        )

        return df_v_t

    def step_r(self, df_clust, df_F):
        """Calculate r for a single timestep t, dt = t - t_0::

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

        df_r_t = df_F.join(df_clust, "id").selectExpr(
            "id",
            f"`Fx`*{self.dt}*{self.dt}/2 + `vx` * {self.dt} + `x` as `x`",
            f"`Fy`*{self.dt}*{self.dt}/2 + `vy` * {self.dt} + `y` as `y`",
            f"`Fz`*{self.dt}*{self.dt}/2 + `vz` * {self.dt} + `z` as `z`",
            "m"
        )

        return df_r_t


class Intergrator_Euler2(Intergrator_Euler):
    """Improved Euler integrator - 2nd order. After calculating F(1) for the current position,
    it calculates provisional coordiantes r(1) (this is like the original so far). 
    It then uses r(1) to calculate the force F(2), and uses the average of F(1) and F(2)
    in the final calculation of coordinates and velocities for the step

    Suggested nparts: # cores / 2
    """

    def advance(self, df_clust):
        """Advance by a step;

        :param insteps: number of steps to advance
        :type insteps: int
        :returns: the new positions and velocities of the particles of the cluster, and time passed
        :rtype: tuple (pyspark.sql.DataFrame, with schema schemas.clust_input, float)
        """

        df_clust = df_clust.localCheckpoint().repartition(self.nparts, "id")
        df_F1 = self.calc_F(df_clust).localCheckpoint().repartition(self.nparts, "id")
        df_r1 = self.step_r(df_clust, df_F1)
        df_r1 = df_r1.localCheckpoint().repartition(self.nparts, "id")

        df_F2 = self.calc_F(df_r1)

        df_F = utils.df_elementwise(df_F1, df_F2, "id", "+", "Fx", "Fy", "Fz")
        df_F = df_F.selectExpr("id",
                               "`Fx` / 2 as `Fx`",
                               "`Fy` / 2 as `Fy`",
                               "`Fz` / 2 as `Fz`"
                               )
        df_F.localCheckpoint()

        df_v, df_r = self.step_v(df_clust, df_F), self.step_r(
            df_clust, df_F)

        df_clust = df_r.join(df_v, "id").join(
            df_clust.select("id", "m"), "id")

        return (df_clust, self.dt)


def calc_gforce_cartesian(df_clust, G=1):
    """Calculate the distance and gravity force between
    every two particles in the cluster::

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
