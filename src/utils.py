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


def load_df(floc, schema=None, header="true", limit=None, part=None, **kwargs):
    """
    Wrapper for pyspark.sql.SparkSession.load - read dataframe from parquet or csv

    :param floc: path to the file(s) - load accepts wildcards
    :type floc: str

    :param schema: schema to use for the datframe, defaults to None
    :type schema: pyspark.sql.types.StructType, optional
    :param header: does the input data have a header, defaults to "true"
    :type header: str {"true"/"false"}, optional
    :param limit: if set reads only the first {limit} rows, defaults to None
    :type limit: int, optional
    :param part: if set repartitions the datafarame by this parameter (pyspark.sql.DataFrame.repartition), defaults to None
    :type part: *str - repartition with the default number of partitions by this column name
                *int - repartition into this number number of partitions, optional
    :param **kwargs: additional arguments to pass to load
    :type **kwargs: dict
    :returns: the resulting dataframe
    :rtype: pyspark.sql.DataFrame
    :raises: ValueError if the file (floc) extension is not 'csv' or 'parquet'
    """
    if floc.endswith("parquet"):
        fformat = "parquet"
    elif floc.endswith("csv") or floc.endswith("csv.gz"):
        fformat = "csv"
    else:
        raise ValueError(
            "can't load data, specify file extension"
            " [parquet, csv] in the filename")

    spark = SparkSession.builder.getOrCreate()

    df = spark.read.load(floc, format=fformat,
                         header=header, schema=schema, **kwargs)

    if limit:
        df = df.limit(limit)
    if part:
        return df.repartition(part)
    return df


def save_df(df, fname, pth, fformat="parquet", compression="gzip", **kwargs):
    """Wrapper for pyspark.sql.DataFrame.save - save a dataframe

    :param df: dataframe to save
    :type df: pyspark.sql.DataFrame
    :param fname: filename
    :type fname: str
    :param pth: path to the folder the file(s) is in
    :type pth: str
    :param **kwargs: additional arguments to pass to save
    :type **kwargs: dict
    :param fformat: format to save in, defaults to "parquet"
    :type fformat: str, optional
    :param compression: compression to use, defaults to "gzip"
    :type compression: str, optional
    :returns: path to the saved dataframe
    :rtype: str
    """
    sc = SparkContext.getOrCreate()
    sloc = os.path.join(pth, "{}-{}-{}".format(fname,
                                               clean_str(sc.appName),
                                               sc.applicationId))
    df.write.format(fformat).save(sloc, compression=compression, **kwargs)

    return sloc


def df_agg_sum(df, aggCol, *sumCols):
    """Dataframe - aggregate by column and sum

    groups all the rows that have the same value for aggCol,
    and sums their sumCols
    :param df: dataframe to use
    :type df: pyspark.sql.DataFrame
    :param aggCol: column to aggrate on
    :type aggCol: str
    :param *sumCols: columns to sum
    :type *sumCols: str
    :returns: resulting dataframe
    :rtype: pyspark.sql.DataFrame
    """
    df_agg = df.groupBy(aggCol).sum(*sumCols)
    renameCols = [f"`sum({col})` as `{col}`" for col in sumCols]
    return df_agg.selectExpr(aggCol, *renameCols)


def df_x_cartesian(df, ffilter=None):
    """Get the cartesian product of a dataframe with itself

    :param df: dataframe to use
    :type df: pyspark.sql.DataFrame
    :param ffilter: SQL string to filter the final product by
    :type ffilter: str
    :returns: resulting dataframe
    :rtype: pyspark.sql.DataFrame
    """
    renameCols = [f"`{col}` as `{col}_other`" for col in df.columns]
    df_cart = df.crossJoin(df.selectExpr(renameCols))
    if ffilter:
        return df_cart.filter(ffilter)
    return df_cart


def df_add_index(df, order_col):
    """Add an index column to a dataframe

    :param df: dataframe to use
    :type df: pyspark.sql.DataFrame
    :param order_col: column to order by
    :type order_col: str
    :returns: resulting dataframe
    :rtype: pyspark.sql.DataFrame
    """
    return df.withColumn('index',
                         row_number().over(Window.orderBy(order_col)) - 1)


def df_elementwise(df, df_other, idCol, op, *cols, renameOutput=False):
    """Join two dataframes with the same schema by id,
    and perform an elementwise operation on the selected columns

    :param df: first dataframe to use
    :type df: pyspark.sql.DataFrame
    :param df_other: second dataframe to use
    :type df_other: pyspark.sql.DataFrame
    :param idCol: column containing the matching ids
    :type idCol: str
    :param op: operation to perform
    :type op: str {"+", "-", "*", "/"}
    :param *cols: columns to perform the operation on
    :type *cols: str
    :param renameOutput: if True, add the opeartion performed to the resulting columns' names, defaults to False
    :type renameOutput: bool, optional
    :returns: resulting dataframe
    :rtype: pyspark.sql.DataFrame
    """

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


def df_join_rename(df, df_other, idCol):
    """Join two dataframes with the same columns and rename the second ones

    :param df: first dataframe to use
    :type df: pyspark.sql.DataFrame
    :param df_other: second dataframe to use
    :type df_other: pyspark.sql.DataFrame
    :param idCol: column containing the matching ids
    :type idCol: str
    """
    renameCols = [f"`{col}` as `{col}_other`" for col in df_other.columns if col != idCol]
    return df.join(df_other.selectExpr(idCol, *renameCols), idCol, "inner")


class NpAccumulatorParam(AccumulatorParam):
    """spark acumulator param for a numpy array"""

    def zero(self, initialValue):
        return np.zeros(initialValue.shape)

    def addInPlace(self, v1, v2):
        v1 += v2
        return v1


"""comparison"""


def df_compare(df, df_other, idCol):
    """Comapre two dataframes with the same schema and ids
    by taking the absolute difference of each row

    :param df: first dataframe to use
    :type df: pyspark.sql.DataFrame
    :param df_other: second dataframe to use
    :type df_other: pyspark.sql.DataFrame
    :param idCol: column containing the matching ids
    :type idCol: str
    :returns: resulting dataframe
    :rtype: pyspark.sql.DataFrame
    """

    cols = df.columns[:]
    cols.remove(idCol)

    df_diff = df_elementwise(df, df_other, idCol, '-', *cols)

    absCols = [f"abs(`{col}`) as `{col}`" for col in cols]

    return df_diff.selectExpr(idCol, *absCols)


def simple_error(df, df_target, idCol):
    """Get the absolute differences between all elements
    of two dataframes with the same schema, and sum them

    note: since inner join is used
    elements not peresent in one of the dataframes are ignored

    :param df: first dataframe to use
    :type df: pyspark.sql.DataFrame
    :param df_target: second dataframe to use
    :type df_target: pyspark.sql.DataFrame
    :param idCol: column containing the matching ids
    :type idCol: str
    :returns: total difference
    :rtype: float
    """

    df_adiff = df_compare(df, df_target, idCol)

    sumCols = df_adiff.drop(idCol).groupBy().sum().collect()[0]

    return sum(sumCols)


def mse(df, df_target, idCol, rmse=False):
    """Get the mean squared error or root mean squared error
    between two dataframes with the same schema

    :param df: first dataframe to use
    :type df: pyspark.sql.DataFrame
    :param df_target: second dataframe to use
    :type df_target: pyspark.sql.DataFrame
    :param idCol: column containing the matching ids
    :type idCol: str
    :param rmse: get the root mean squared error instead, defaults to False
    :type rmse: bool, optional
    :returns: [r]mse
    :rtype: float
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
    """Collect from a dataframe up to a limit

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


def plot_cluster_scater2d(df_clust, axes=["x", "y"], title="Plot", stack=False, fout=None, eqAspect=True):
    """Draw a 2d scatter plot of a cluster

    :param df_clust: dataframe containing the cluster
    :type df_clust: pyspark.sql.DataFrame
    :param axes: axes to plot
    :type axes: list, optional
    :param title: title of the plot, defaults to "Plot"
    :type title: str, optional
    :param stack: don't show/save the plot, for drawing continuously on the same plot
    :type stack: bool, optional
    :param fout: save the plot to a file if provided, defaults to None
    :type fout: str, optional
    :param eqAspect: should it use set_aspect('equal') 
    :type eqAspect: bool, optional
    """

    coords = np.transpose(df_clust.select(*axes).collect())

    pl.scatter(coords[0], coords[1], s=1,
               label="Data set", **{"facecolors": "blue", "marker": "."})

    pl.xlabel(axes[0])
    pl.ylabel(axes[1])
    if eqAspect:
        pl.axes().set_aspect('equal')

    pl.title(title)
    if not stack:
        if fout is not None:
            pl.savefig(fout)
        else:
            pl.show()


def plot_cluster_scater3d(df_clust, title="Plot", fout=None):
    """Draw a 3d scatter plot of a cluster

    :param df_clust: dataframe containing the cluster
    :type df_clust: pyspark.sql.DataFrame
    :param title: title of the plot, defaults to "Plot"
    :type title: str, optional
    :param fout: save the plot to a file if provided, defaults to None
    :type fout: str, optional
    """
    coords = np.transpose(df_clust.select("x", "y", "z").collect())

    scatter3d_mpl(coords[0], coords[1], coords[2],
                  label="Data set", **{"facecolors": "blue", "marker": "."})

    pl.title(title)
    if fout is not None:
        pl.savefig(fout)
    pl.show()


def plot_histogram(df, col, title="Plot", fout=None):
    """Plot a single dataframe column as a histogram

    :param df_clust: dataframe to use
    :type df: pyspark.sql.DataFrame
    :param col: column to plot the histogram for
    :type col: str
    :param title: title of the plot, defaults to "Plot"
    :type title: str, optional
    :param fout: save the plot to a file if provided, defaults to None
    :type fout: str, optional
    """
    arr_gfs = np.array(df.select(col).collect())\
        .transpose()[0]  # collect returns a list of lists

    pl.hist(arr_gfs, rwidth=0.8, bins="auto")

    pl.show()


"""misc"""


def clean_str(string):
    """Clean a string from everything except word characters,
    replace spaces with '_'

    :param string: string to clean
    :type string: str
    :returns: result
    :rtype: str
    """
    string = re.sub(r"\s", "_", string.strip())
    return re.sub(r"[^\w]", "", string)


class SaveOptions(dict):
    """Configuration for the utils.save_df method that can be easily reused::

        save_df(df, fname, **SaveOptions)

    :param pth: path to the folder the file(s) is in
    :type pth: str
    :param **kwargs: additional arguments to pass to save
    :type **kwargs: dict
    :param fformat: format to save in, defaults to "parquet"
    :type fformat: str, optional
    :param compression: compression to use, defaults to "gzip"
    :type compression: str, optional
    """
    __slots__ = ()
    def __init__(self, pth, fformat="parquet", compression="gzip", **kwargs):

        super(SaveOptions, self).__init__(**kwargs)
        self["pth"] = pth
        self["fformat"] = fformat
        self["compression"] = compression
