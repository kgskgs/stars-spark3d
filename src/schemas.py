#!/usr/bin/python3
from pyspark.sql.types import *

clust_input = StructType([
    StructField('x',  DoubleType(), False),
    StructField('y',  DoubleType(), False),
    StructField('z',  DoubleType(), False),
    StructField('vx', DoubleType(), False),
    StructField('vy', DoubleType(), False),
    StructField('vz', DoubleType(), False),
    StructField('m',  DoubleType(), False),
    StructField('id', IntegerType(), False)
])

"""SPLIT"""
dist_gforce = StructType([
    StructField("dist",   DoubleType(), False),
    StructField("gforce", DoubleType(), False),
    StructField("gx",     DoubleType(), False),
    StructField("gy",     DoubleType(), False),
    StructField("gz",     DoubleType(), False)
])

dist_gforce_cartesian = StructType([
    StructField('id',       IntegerType(), False),
    StructField('id_other', IntegerType(), False),
    StructField("dist",     DoubleType(), False),
    StructField("gforce",   DoubleType(), False),
    StructField("gx",       DoubleType(), False),
    StructField("gy",       DoubleType(), False),
    StructField("gz",       DoubleType(), False)
])

gforce_id = StructType([
    StructField('id',     IntegerType(), False),
    StructField("gforce", DoubleType(), False),
    StructField("gx",     DoubleType(), False),
    StructField("gy",     DoubleType(), False),
    StructField("gz",     DoubleType(), False)
])


F = StructType([
    StructField("Fx", DoubleType(), False),
    StructField("Fy", DoubleType(), False),
    StructField("Fz", DoubleType(), False)
])

F_cartesian = StructType([
    StructField('id',       IntegerType(), False),
    StructField('id_other', IntegerType(), False),
    StructField("Fx",       DoubleType(), False),
    StructField("Fy",       DoubleType(), False),
    StructField("Fz",       DoubleType(), False)
])


F_id = StructType([
    StructField('id', IntegerType(), False),
    StructField("Fx", DoubleType(), False),
    StructField("Fy", DoubleType(), False),
    StructField("Fz", DoubleType(), False)
])


r_id = StructType([
    StructField('id', IntegerType(), False),
    StructField("x",  DoubleType(), False),
    StructField("y",  DoubleType(), False),
    StructField("z",  DoubleType(), False)
])

v_id = StructType([
    StructField('id', IntegerType(), False),
    StructField("vx", DoubleType(), False),
    StructField("vy", DoubleType(), False),
    StructField("vz", DoubleType(), False)
])

"""Test"""


E_test_res = StructType([
    StructField('name',       StringType(), False),
    StructField("E",          DoubleType(), False),
    StructField("targetE",    DoubleType(), False),
    StructField("difference", DoubleType(), False),
    StructField("tolerance",   DoubleType(), False),
    StructField("success",    BooleanType(), False),
])