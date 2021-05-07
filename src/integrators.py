import utils
from integrator_base import IntegratorBase


class IntergratorEuler(IntegratorBase):
    """First order integration method that advances all particles
    at a constant time step (dt). Calculates effective force per unit mass
    for each particle with O(n^2) complexity.
    """

    def advance(self, df_clust):
        """Advance by a step

        :param df_clust: cluster data - position, velocity, and mass
        :type df_clust: pyspark.sql.DataFrame, with schema schemas.clust
        :returns: the new positions and velocities of the particles of the cluster, and time passed
        :rtype: tuple (pyspark.sql.DataFrame, with schema schemas.clust, float)
        """

        df_clust = df_clust.cache()
        df_F = self.calc_F(df_clust)

        df_clust = self.step(df_clust, df_F)

        return (df_clust, self.dt)

    def step_v(self, df_clust, df_F):
        """Calculate v for a single timestep t, dt = t - t_0::

            v_i(t) = F_i*∆t + v_i(t0)

        :param df_clust: cluster data - position, velocity, and mass
        :type df_clust: pyspark.sql.DataFrame, with schema schemas.clust
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

        :param df_clust: cluster data - position, velocity, and mass
        :type df_clust: pyspark.sql.DataFrame, with schema schemas.clust
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


    def step(self, df_clust, df_F):
        """combination of setp_r and setp_v for more efficient computation

        :param df_clust: cluster data - position, velocity, and mass
        :type df_clust: pyspark.sql.DataFrame, with schema schemas.clust
        :param df_F: force per unit of mass acting on each particle
        :type df_F: pyspark.sql.DataFrame, with schema schemas.F_id
        :returns: the new velocity of each particle
        :rtype: pyspark.sql.DataFrame, with schema schemas.clust
        """
        df_clust = df_F.join(df_clust, "id").selectExpr(
            "id",
            f"`Fx`*{self.dt}*{self.dt}/2 + `vx` * {self.dt} + `x` as `x`",
            f"`Fy`*{self.dt}*{self.dt}/2 + `vy` * {self.dt} + `y` as `y`",
            f"`Fz`*{self.dt}*{self.dt}/2 + `vz` * {self.dt} + `z` as `z`",
            f"`Fx`*{self.dt} + `vx` as `vx`",
            f"`Fy`*{self.dt} + `vy` as `vy`",
            f"`Fz`*{self.dt} + `vz` as `vz`",
            "m"
        )

        return df_clust


class IntergratorEuler2(IntergratorEuler):
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
        :rtype: tuple (pyspark.sql.DataFrame, with schema schemas.clust, float)
        """

        df_F1 = self.calc_F(df_clust)
        df_F1 = df_F1.cache()

        df_r1 = self.step_r(df_clust, df_F1)

        df_F2 = self.calc_F(df_r1)

        df_F = utils.df_elementwise(df_F1, df_F2, "id", "+", "Fx", "Fy", "Fz")
        df_F = df_F.selectExpr("id",
                               "`Fx` / 2 as `Fx`",
                               "`Fy` / 2 as `Fy`",
                               "`Fz` / 2 as `Fz`"
                               )

        df_clust = self.step(df_clust, df_F)

        return (df_clust, self.dt)


class IntegratorRungeKutta4(IntegratorBase):
    """Fourth order Runge-Kutta integrator. Splits the interval ∆t in four stages.
    """

    def advance(self, df_clust):
        """advance by a step::

            x_n+1= x_n + ∆t/6 * (k1 + 2*k2 + 2*k3 + k4)
            k1 = f(t_n, x_n);
            k2 = f(t_n + ∆t/2, x_n + k1*∆t/2);
            k3 = f(t_n + ∆t/2, x_n + k2*∆t/2);
            k4 = f(t_n + ∆t, x_n + k3*∆t)

        :param df_clust: cluster data
        :type df_clust: pyspark.sql.DataFrame, with schema schemas.clust
        :returns: the new positions and velocities of the particles of the cluster, and time passed
        :rtype: tuple (pyspark.sql.DataFrame, with schema schemas.clust, float)
        """

        F1 = self.calc_F(df_clust)
        F1 = F1.cache()
        F2 = self.calc_F(self.step(df_clust, F1, 0.5))
        F2 = F2.cache()
        F3 = self.calc_F(self.step(df_clust, F2, 0.5))
        F3 = F3.cache()
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
        """Calculate new positions & velocities from F with a modifier added to ∆t

        :param df_clust: cluster data
        :type df_clust: pyspark.sql.DataFrame, with schema schemas.clust
        :param df_F: force per unit of mass acting on each particle
        :type df_F: pyspark.sql.DataFrame, with schema schemas.F_id
        :param mod: modifier to apply to ∆t, defaults to 1
        :type mod: double, optional
        :returns: cluster data after force is applied
        :rtype: pyspark.sql.DataFrame, with schema schemas.clust
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


class IntegratorLeapfrog(IntegratorBase):
    """Second order leapfrog integrator
    """

    def __init__(self, dt, G=1):
        """Constructor"""
        super().__init__(dt, G)
        self.df_F_t0 = None

    def advance(self, df_clust):
        """advance by a step

        :param df_clust: cluster data
        :type df_clust: pyspark.sql.DataFrame, with schema schemas.clust
        :returns: the new positions and velocities of the particles of the cluster, and time passed
        :rtype: tuple (pyspark.sql.DataFrame, with schema schemas.clust, float)
        """

        if not self.df_F_t0:
            self.df_F_t0 = self.calc_F(df_clust)

        df_r = self.step_r(df_clust)
        df_F_t = self.calc_F(df_r)
        df_F_t = df_F_t.localCheckpoint()
        df_v = self.step_v(df_clust, df_F_t)

        self.df_F_t0 = df_F_t

        df_clust = df_r.join(df_v, "id")
        # bring order back to schema
        df_clust = df_clust.select('id', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'm')

        return (df_clust, self.dt)

    def step_r(self, df_clust):
        """Calculate r

        :param df_clust: cluster data - position, velocity, and mass
        :type df_clust: pyspark.sql.DataFrame, with schema schemas.clust
        :returns: the new position of each particle
        :rtype: pyspark.sql.DataFrame, with schema schemas.r_id
        """

        df_r_t = self.df_F_t0.join(df_clust, "id").selectExpr(
            "id",
            f"`Fx`*{self.dt}*{self.dt}/2 + `vx` * {self.dt} + `x` as `x`",
            f"`Fy`*{self.dt}*{self.dt}/2 + `vy` * {self.dt} + `y` as `y`",
            f"`Fz`*{self.dt}*{self.dt}/2 + `vz` * {self.dt} + `z` as `z`",
            "m"
        )

        return df_r_t

    def step_v(self, df_clust, df_F_t):
        """Calculate v based on the average of two forces::

            v_i(t) = 1/2(F_i(t0) + F_i(t))*∆t + v_i(t0)

        :param df_clust: cluster data - position, velocity, and mass
        :type df_clust: pyspark.sql.DataFrame, with schema schemas.clust
        :param df_F_t: force per unit of mass acting on each particle
        :type df_F_t: pyspark.sql.DataFrame, with schema schemas.F_id
        :returns: the new position of each particle
        :rtype: pyspark.sql.DataFrame, with schema schemas.r_id
        """

        df_F = utils.df_elementwise(self.df_F_t0, df_F_t, "id", "+", "Fx", "Fy", "Fz")

        df_v_t = df_F.join(df_clust, "id").selectExpr(
            "id",
            f"`Fx`*{self.dt}/2 + `vx` as `vx`",
            f"`Fy`*{self.dt}/2 + `vy` as `vy`",
            f"`Fz`*{self.dt}/2 + `vz` as `vz`"
        )

        return df_v_t
