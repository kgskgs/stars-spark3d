#!/usr/bin/python3
import numpy as np
import os
from math import sqrt
from scipy.constants import G
from matplotlib import pyplot as pl
import re

from pyspark.sql.session import SparkSession  
from pyspark.sql.functions import row_number
from pyspark.sql import Window
from pyspark import AccumulatorParam



def load_cluster_data(fname, pth, schema=None, header="true", limit=None, part=None):
    """read dataframe from csv"""
    spark = SparkSession.builder.getOrCreate()

    floc = os.path.join(pth, fname)

    df = spark.read.load(floc, format="csv", header=header, schema=schema)

    if limit:
        df = df.limit(limit)
    if part:
        return df.repartition(part)
    return df


def plot_cluster_scater(df_clust, title="Plot", fout=None):
    """draw a scatter plot of a cluster"""
    coords = np.transpose(df_clust.select("x", "y", "z").collect())

    scatter3d_mpl(coords[0], coords[1], coords[2],label="Data set", **{"facecolors":"blue", "marker": "."})

    pl.title(title)
    if fout != None:
        pl.savefig(fnout)
    pl.show()

def plot_histogram(df, col, title="Plot", fout=None):
    """Plot a single dataframe column as a histogram"""
    arr_gfs = np.array(df.select(col).collect())\
            .transpose()[0] #collect returns a list of lists

    pl.hist(arr_gfs, rwidth=0.8, bins="auto")

    pl.show()


class NpAccumulatorParam(AccumulatorParam):
    """spark acumulator param for a numpy array"""
    def zero(self, initialValue):
        return np.zeros(initialValue.shape)

    def addInPlace(self, v1, v2):
        v1 += v2
        return v1


def df_add_index(df, order_col):
    """add an index column to a dataframe"""
    return df.withColumn('index', 
        row_number().over(Window.orderBy(order_col))-1)

def clean_str(string):
    """clean a string from everything except word characters, replace spaces with '_'
    """
    string = re.sub(r"\s", "_", string.strip())
    return re.sub(r"[^\w]", "", string)

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
    return normv(vec, dist) * G*mass1*mass2/dist**2

