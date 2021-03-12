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
        :type df_clust: pyspark.sql.DataFrame, with schema schemas.clust_input
        :returns: the new positions and velocities of the particles of the cluster, and time passed
        :rtype: tuple (pyspark.sql.DataFrame, with schema schemas.clust_input, float)
        """

        # df_clust = df_clust.localCheckpoint()
        df_F = self.calc_F(df_clust)
        df_F = df_F.localCheckpoint()
        df_v, df_r = self.step_v(df_clust, df_F), self.step_r(
            df_clust, df_F)

        df_clust = df_r.join(df_v, "id")
        # bring order back to schema
        df_clust = df_clust.select('id', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'm')

        return (df_clust, self.dt)


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