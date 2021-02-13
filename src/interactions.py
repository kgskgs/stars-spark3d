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

        self.spark = SparkSession.builder.getOrCreate()

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

            self.cluster = self.cluster.localCheckpoint()

            if self.dt_out and self.next_out <= self.t:
                self.snapshot()
                self.next_out += self.dt_out
            if self.dt_diag and self.next_diag <= self.t:
                self.diag()
                self.next_diag += self.dt_diag

    def snapshot(self, add_t=True):
        """Save a snapshot of the cluster

        :param add_t: if true add timestamp to each particle on output
        :type add_t: bool
        """
        if add_t:
            utils.save_df(self.cluster.withColumn("t", lit(float(self.t))),
                          f"t{self.t}", **self.save_params)
        else:
            utils.save_df(self.cluster, f"t{self.t}", **self.save_params)

    def diag(self):
        """Save diagnostic information about the cluster energy
        """

        T, U = cluster.calc_T(self.cluster, self.G), cluster.calc_U(self.cluster, self.G)
        E = T + U
        cm = cluster.calc_cm(self.cluster)

        df_diag = self.spark.createDataFrame(
            [(self.t, cm['x'], cm['y'], cm['z'], T, U, E)],
            schema=schemas.diag
        )

        utils.save_df(df_diag, f"diag_t{self.t}", **self.save_params)


class Intergrator_Euler:
    """First order integration method that advances all particles
    at a constant time step (dt). Calculates effective force per unit mass
    for each particle with O(n^2) complexity.

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

        # df_clust = df_clust.repartition(self.nparts, "id")
        # df_clust = df_clust.localCheckpoint()
        df_F = self.calc_F(df_clust)
        df_F = df_F.localCheckpoint()
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
    """

    def advance(self, df_clust):
        """Advance by a step;

        :param insteps: number of steps to advance
        :type insteps: int
        :returns: the new positions and velocities of the particles of the cluster, and time passed
        :rtype: tuple (pyspark.sql.DataFrame, with schema schemas.clust_input, float)
        """

        # df_clust = df_clust.repartition(self.nparts, "id")
        df_clust = df_clust.localCheckpoint()

        df_F1 = self.calc_F(df_clust)  # .repartition(self.nparts, "id")
        df_F1 = df_F1.localCheckpoint()

        df_r1 = self.step_r(df_clust, df_F1)  # .repartition(self.nparts, "id")
        df_r1 = df_r1.localCheckpoint()

        df_F2 = self.calc_F(df_r1)
        df_F2 = df_F2.localCheckpoint()

        df_F = utils.df_elementwise(df_F1, df_F2, "id", "+", "Fx", "Fy", "Fz")
        df_F = df_F.selectExpr("id",
                               "`Fx` / 2 as `Fx`",
                               "`Fy` / 2 as `Fy`",
                               "`Fz` / 2 as `Fz`"
                               )
        df_F = df_F.localCheckpoint()

        df_v, df_r = self.step_v(df_clust, df_F), self.step_r(
            df_clust, df_F)

        df_clust = df_r.join(df_v, "id")
        # bring order back to schema
        df_clust = df_clust.select('id', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'm')

        return (df_clust, self.dt)


class Integrator_RungeKutta4(Intergrator_Euler):
    """Fourth order Runge-Kutta integrator. Splits the interval ∆t in four stages.
    """

    def advance(self, df_clust):
        """advance by a step::

            x_n+1= x_n + ∆t/6 * (k1 + 2*k2 + 2*k3 + k4)
            k1 = f(t_n, x_n);
            k2 = f(t_n + ∆t/2, x_n + k1*∆t/2);
            k3 = f(t_n + ∆t/2, x_n + k2*∆t/2);
            k4 = f(t_n + ∆t, x_n + k3*∆t)

        [Roa J. et al. (2020). Moving Planets Around
        eq. (5.11), (5.12)]

        :param df_clust: cluster data
        :type df_clust: pyspark.sql.DataFrame, with schema schemas.clust_input
        :returns: the new positions and velocities of the particles of the cluster, and time passed
        :rtype: tuple (pyspark.sql.DataFrame, with schema schemas.clust_input, float)
        """

        F1 = self.calc_F(df_clust)
        F1 = F1.localCheckpoint()
        F2 = self.calc_F(self.step(df_clust, F1, 0.5))
        F2 = F2.localCheckpoint()
        F3 = self.calc_F(self.step(df_clust, F2, 0.5))
        F3.localCheckpoint()
        F4 = self.calc_F(self.step(df_clust, F3))

        F = utils.df_elementwise(F1,
                utils.df_elementwise(F2.selectExpr("id", "`Fx` * 2 as `Fx`", "`Fy` * 2 as `Fy`", "`Fz` * 2 as `Fz`"),
                    utils.df_elementwise(F3.selectExpr("id", "`Fx` * 2 as `Fx`", "`Fy` * 2 as `Fy`", "`Fz` * 2 as `Fz`"),
                        F4,
                    "id", "+", "Fx", "Fy", "Fz"),
                "id", "+", "Fx", "Fy", "Fz"),
            "id", "+", "Fx", "Fy", "Fz")

        F = F.selectExpr("id", "Fx / 6 as Fx", "Fy / 6 as Fy", "Fz / 6 as Fz")

        df_clust = self.step(df_clust, F)

        return (df_clust, self.dt)

    def step(self, df_clust, df_F, mod=1):
        """calculate new positions & velocities from F with a modifier added to ∆t

        :param df_clust: cluster data
        :type df_clust: pyspark.sql.DataFrame, with schema schemas.clust_input
        :param df_F: force per unit of mass acting on each particle
        :type df_F: pyspark.sql.DataFrame, with schema schemas.F_id
        :param mod: modifier to apply to ∆t, defaults to 1
        :type mod: double, optional
        :returns: cluster data after force is applied
        :rtype: pyspark.sql.DataFrame, with schema schemas.clust_input
        """

        df_clust = df_F.join(df_clust, "id").selectExpr(
            "id",
            f"(`Fx`*{self.dt}*{self.dt}*{mod}*{mod}/2 + `vx` * {self.dt} * {mod}) + `x` as `x`",
            f"(`Fy`*{self.dt}*{self.dt}*{mod}*{mod}/2 + `vy` * {self.dt} * {mod}) + `y` as `y`",
            f"(`Fz`*{self.dt}*{self.dt}*{mod}*{mod}/2 + `vz` * {self.dt} * {mod}) + `z` as `z`",
            f"(`Fx`*{self.dt}) * {mod} + `vx` as `vx`",
            f"(`Fy`*{self.dt}) * {mod} + `vy` as `vy`",
            f"(`Fz`*{self.dt}) * {mod} + `vz` as `vz`",
            "m"
        )

        return df_clust
