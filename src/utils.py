#!/usr/bin/python3
import numpy as np
import os
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
    """wrapper - read dataframe from parquet or csv"""
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
    """wrapper - save a dataframe"""
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


def df_x_cartesian(df, ffilter=None):
    """
    get the cartesian product of a dataframe with itself

    :param str ffilter: SQL string to filter the final product by
    """
    renameCols = [f"`{col}` as `{col}_other`" for col in df.columns]
    df_cart = df.crossJoin(df.selectExpr(renameCols))
    if ffilter:
        return df_cart.filter(ffilter)
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


"""comparison"""


def df_compare(df, df_other, idCol):
    """comapre two dataframes with the same schema and ids
    by taking the absolute difference of each row"""

    cols = df.columns[:]
    cols.remove(idCol)

    df_diff = df_elementwise(df, df_other, idCol, '-', *cols)

    absCols = [f"abs(`{col}`) as `{col}`" for col in cols]

    return df_diff.selectExpr(idCol, *absCols)


def simple_error(df, df_target, idCol):
    """get the absolute differences between all elements
    of two dataframes with the same schema, and sum them

    note: since inner join is used
    elements not peresent in one of the dataframes are ignored
    """

    df_adiff = df_compare(df, df_target, idCol)

    sumCols = df_adiff.drop(idCol).groupBy().sum().collect()[0]

    return sum(sumCols)


def mse(df, df_target, idCol, rmse=False):
    """
    get the mean squared error or root mean squared error
    between two dataframes with the same schema
    """
    cols = df.columns[:]
    cols.remove(idCol)

    df_diff = df_elementwise(df, df_target, idCol, '-', *cols).drop(idCol)
    n = df_diff.count()

    powCols = [f"`{col}` * `{col}` as `{col}`" for col in cols]
    df_mse = df_diff.selectExpr(*powCols)
    df_mse = df_mse.groupBy().sum()

    if rmse:
        meanCols = [f"sqrt(`sum({col})` / {n}) as `rmse({col})`"
                    for col in cols]
    else:
        meanCols = [f"`sum({col})` / {n} as `mse({col})`"
                    for col in cols]

    return df_mse.selectExpr(*meanCols)


def df_collectLimit(df, limit, *cols, sortCol=None):
    """
    Collect from a dataframe up to a limit

    :param df: dataframe to collect
    :type df: pyspark.sql.DataFrame
    :param limit: maximum number of rows to collect
    :type limit: int
    :param *cols: columns to collect
    :type *cols: str
    :param sortCol: column to sort by (so you always get the same rows if needed)
    :type sortCol: str
    :returns: collected columns of df if col specified or all the columns
    :rtype: list
    """
    if sortCol:
        df = df.sort(sortCol)

    if df.count() > limit:
        df = df.limit(limit)

    if cols:
        return df.select(*cols).collect()
    return df.collect()


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
