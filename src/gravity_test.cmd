del /Q D:\tmp\spark-events\*

call start spark-submit --name "7 - gravity_udf.py" src/gravity_udf.py --limit 7
call start spark-submit --name "7 - gravity_rdds.py" src/gravity_rdds.py --limit 7
call start spark-submit --name "7 - gravity_accumulator.py" src/gravity_accumulator.py --limit 7