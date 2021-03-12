import utils
from abc import ABC, abstractmethod


class IntegratorBase(ABC):
    """abstract class, contains the force calculation
    that is used by all integrators

    :param dt: time step for the simulation
    :type dt: float
    :param G: gravitational constant to use, defaults to 1
    :type G: float, optional
    """

    def __init__(self, dt, G=1):
        """Constructor"""

        self.dt = dt

        self.G = G

    @abstractmethod
    def advance(self, df_clust):
        """Advance by a step

        Has to return the same value across all implemetations as it's used in Simulation
        
        :param df_clust: cluster data - position, velocity, and mass
        :type df_clust: pyspark.sql.DataFrame, with schema schemas.clust_input
        :returns: the new positions and velocities of the particles of the cluster, and time passed
        :rtype: tuple (pyspark.sql.DataFrame, with schema schemas.clust_input, float)
        """
        pass

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