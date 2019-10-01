#!/usr/bin/python3

from pyspark.sql.types import *
from pyspark3d.visualisation import scatter3d_mpl
import pylab as pl
import numpy as np
from math import sqrt


from pyspark.sql.functions import row_number, monotonically_increasing_id
from pyspark.sql import Window
from pyspark import AccumulatorParam

from scipy.constants import G

class SparkUtils():
    """utils that require a spark session"""
    def __init__(self, spark_session):
            self.spark = spark_session

    def load_cluster_data(self, fname, pth="../data/", header="true", limit=None):
        """load a cluster from the dataset
        https://www.kaggle.com/mariopasquato/star-cluster-simulations 
        """
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

        if limit:
            return df.limit(limit)
        return df


    def plot_cluster_scater(self, df_clust, title, fout=None):
        """draw a scatter plot of a cluster"""
        coords = np.transpose(df_clust.select("x", "y", "z").collect())

        scatter3d_mpl(coords[0], coords[1], coords[2],label="Data set", **{"facecolors":"blue", "marker": "."})

        pl.title(title)
        if fout != None:
            pl.savefig(fnout)
        pl.show()

class NpAccumulatorParam(AccumulatorParam):
    def zero(self, initialValue):
        return np.zeros(initialValue.shape)

    def addInPlace(self, v1, v2):
        v1 += v2
        return v1


def df_add_index(df, order_col):
    """add an index column to a dataframe"""
    #df = df.rdd.zipWithIndex()
    #return df.toDF()
    return df.withColumn('index', 
        row_number().over(Window.orderBy(order_col))-1)

def vlen3d(v):
    """calucalte the length of a 3d vector"""
    return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

def ptv(p1, p2):
    """get a vector from start and end ponts"""
    return p2 - p1

def normv(v, length=None):
    """normalize a vector"""
    if length == None:
        length = vlen3d(v)
    return v/length


def get_gforce(coords1, coords2, mass1, mass2):
    """calculate gravitational force between two points"""
    if all(coords1 == coords2):
        return np.zeros(3)
    vec = ptv(coords1, coords2)
    dist = vlen3d(vec)
    return normv(vec) * G*mass1*mass2/dist**2


if __name__=="__main__":
    from pyspark.context import SparkContext
    from pyspark.sql.session import SparkSession  

    sc = SparkContext('local')
    spark = SparkSession(sc)

    utils = StarsUtils(spark)


    df = utils.load_cluster_data("c_0000.csv")

    utils.plot_cluster_scater(df, "c_0000")
