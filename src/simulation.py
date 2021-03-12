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
    :type integrator: integrator_base.Integrator_
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
