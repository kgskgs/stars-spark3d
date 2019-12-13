#!/usr/bin/python3
import numpy as np
import os
from math import sqrt
from scipy.constants import G
from matplotlib import pyplot as pl
import re

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession  
from pyspark.sql.functions import row_number
from pyspark.sql import Window
from pyspark import AccumulatorParam

from pyspark3d.visualisation import scatter3d_mpl


def load_df(fname, pth, schema=None, header="true", limit=None, part=None, **kwargs):
    """read dataframe from parquet or csv"""
    spark = SparkSession.builder.getOrCreate()

    if "parquet" in fname:
        fformat = "parquet" 
    elif "csv" in fname:
        fformat = "csv"
    else:
        raise ValueError("can't load data, specify file extension [parquet, csv] in the filename")


    floc = os.path.join(pth, fname)

    df = spark.read.load(floc, format=fformat, header=header, schema=schema, **kwargs)

    if limit:
        df = df.limit(limit)
    if part:
        return df.repartition(part)
    return df

def save_df(df, fname, pth, fformat="parquet", compression="gzip", **kwargs):
    """save a dataframe"""
    sc = SparkContext.getOrCreate()
    sloc = os.path.join(pth, "{}-{}-{}".format(fname, clean_str(sc.appName),sc.applicationId))
    df.write.format(fformat).save(sloc, compression=compression, **kwargs)


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

def plot_relation_2d(df, col1, col2, title="Plot", fout=None):
    """plot a scatter of two columns"""
    pass


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

""
def get_gforce(coords1, coords2, mass1, mass2):
    """calculate gravitational force between two points"""
    if all(coords1 == coords2):
        return np.zeros(3)
    vec = ptv(coords1, coords2)
    dist = vlen3d(vec)
    return normv(vec, dist) * G*mass1*mass2/dist**2

