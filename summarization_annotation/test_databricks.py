from databricks import _get_spark_context
if __name__ == "__main__":

    print("RUNNING SPARK")
    assert _get_spark_context().parallelize([1, 2, 3]).reduce(sum) == 6
    print("DONE RUNNING!! ")