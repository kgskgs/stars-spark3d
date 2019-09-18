#!/usr/bin/python3
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *

from pyspark3d.repartitioning import prePartition
from pyspark3d.repartitioning import repartitionByCol
from pyspark3d.visualisation import scatter3d_mpl

import pylab as pl
import numpy as np



class StarsUtils():

    def __init__(self, spark_session):
        self.spark = spark_session

    def load_cluster_data(self, fname, pth="../data/", header="true"):

        schm = StructType([StructField('x', DoubleType(), True),
                       StructField('y', DoubleType(), True),
                       StructField('z', DoubleType(), True),
                       StructField('vx', DoubleType(), True),
                       StructField('vy', DoubleType(), True),
                       StructField('vz', DoubleType(), True),
                       StructField('m', DoubleType(), True),
                       StructField('id', IntegerType(), True)])

        df = self.spark.read.load(pth + fname, 
                        format="csv", header=header, schema=schm)

        return df

    def plot_cluster_scater(self, df_clust, title, fout=None):
        coords = np.transpose(df_clust.select("x", "y", "z").collect())

        scatter3d_mpl(coords[0], coords[1], coords[2],label="Data set", **{"facecolors":"blue", "marker": "."})

        pl.title(title)
        if fout != None:
            pl.savefig(fnout)
        pl.show()



if __name__=="__main__":

    sc = SparkContext('local')
    spark = SparkSession(sc)

    utils = StarsUtils(spark)


    df = utils.load_cluster_data("c_0000.csv")

    utils.plot_cluster_scater(df, "c_0000")
