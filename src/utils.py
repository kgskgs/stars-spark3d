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

"""spark"""


def load_df(fname, pth, schema=None, header="true", limit=None, part=None, **kwargs):
    """read dataframe from parquet or csv"""
    spark = SparkSession.builder.getOrCreate()

    if "parquet" in fname:
        fformat = "parquet"
    elif "csv" in fname:
        fformat = "csv"
    else:
        raise ValueError(
            "can't load data, specify file extension"
            " [parquet, csv] in the filename")

    floc = os.path.join(pth, fname)

    df = spark.read.load(floc, format=fformat,
                         header=header, schema=schema, **kwargs)

    if limit:
        df = df.limit(limit)
    if part:
        return df.repartition(part)
    return df


def save_df(df, fname, pth, fformat="parquet", compression="gzip", **kwargs):
    """save a dataframe"""
    sc = SparkContext.getOrCreate()
    sloc = os.path.join(pth, "{}-{}-{}".format(fname,
                                               clean_str(sc.appName),
                                               sc.applicationId))
    df.write.format(fformat).save(sloc, compression=compression, **kwargs)

    return sloc


def df_agg_sum(df, aggCol, *sumCols):
    """dataframe - aggregate by column and sum"""
    df_agg = df.groupBy(aggCol).sum(*sumCols)
    renameCols = [f"`sum({col})` as `{col}`" for col in sumCols]
    return df_agg.selectExpr(aggCol, *renameCols)


def df_x_cartesian(df, filterCol=None):
    """get the cartesian product of a dataframe with itself"""
    renameCols = [f"`{col}` as `{col}_other`" for col in df.columns]
    df_cart = df.crossJoin(df.selectExpr(renameCols))
    if filterCol:
        return df_cart.filter(f"{filterCol} != {filterCol}_other")
    return df_cart


def df_add_index(df, order_col):
    """add an index column to a dataframe"""
    return df.withColumn('index',
                         row_number().over(Window.orderBy(order_col)) - 1)


def df_elementwise(df, df_other, idCol, op, *cols, renameOutput=False):
    """join two dataframes with the same schema by id,
    and perform an elementwise operation on the
    selected columns"""

    opStr = {"+": "sum", "-": "dif", "*": "mul", "/": "div"}
    

    renameCols = [f"`{col}` as `{col}_other`" for col in cols]
    df_j = df.join(df_other.selectExpr(idCol, *renameCols), idCol, "inner")

    if renameOutput:
        opCols = [f"`{col}` {op} `{col}_other` as `{opStr[op]}({col})`"
                  for col in cols]
    else:
        opCols = [f"`{col}` {op} `{col}_other` as `{col}`"
                  for col in cols]
    return df_j.selectExpr(idCol, *opCols)


class NpAccumulatorParam(AccumulatorParam):
    """spark acumulator param for a numpy array"""

    def zero(self, initialValue):
        return np.zeros(initialValue.shape)

    def addInPlace(self, v1, v2):
        v1 += v2
        return v1


"""cluster"""


def calc_cm(df_clust):
    """
    calcuate the center of mass of the cluster
    in our dataset all the masses are equal, so
    it is equal to the mean of the coordinates
    """
    df_cm = df_clust.selectExpr("mean(`x`) as `x`",
                                "mean(`y`) as `y`",
                                "mean(`z`) as `z`")

    return df_cm.collect()[0]


"""plots"""


def plot_cluster_scater(df_clust, title="Plot", fout=None):
    """draw a scatter plot of a cluster"""
    coords = np.transpose(df_clust.select("x", "y", "z").collect())

    scatter3d_mpl(coords[0], coords[1], coords[2],
                  label="Data set", **{"facecolors": "blue", "marker": "."})

    pl.title(title)
    if fout is not None:
        pl.savefig(fout)
    pl.show()


def plot_histogram(df, col, title="Plot", fout=None):
    """Plot a single dataframe column as a histogram"""
    arr_gfs = np.array(df.select(col).collect())\
        .transpose()[0]  # collect returns a list of lists

    pl.hist(arr_gfs, rwidth=0.8, bins="auto")

    pl.show()


"""misc"""


def clean_str(string):
    """clean a string from everything except word characters,
    replace spaces with '_'"""
    string = re.sub(r"\s", "_", string.strip())
    return re.sub(r"[^\w]", "", string)


def vlen3d(v):
    """calucalte the length of a 3d vector"""
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def ptv(p1, p2):
    """get a vector from start and end ponts"""
    return p2 - p1


def normv(v, length=None):
    """normalize a vector"""
    if length is not None:
        length = vlen3d(v)
    return v / length


def get_gforce(coords1, coords2, mass1, mass2):
    """calculate gravitational force between two points"""
    if all(coords1 == coords2):
        return np.zeros(3)
    vec = ptv(coords1, coords2)
    dist = vlen3d(vec)
    return normv(vec, dist) * G * mass1 * mass2 / dist**2
