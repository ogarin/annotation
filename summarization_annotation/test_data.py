from databricks import load_metadata
if __name__ == "__main__":
    print("RUNNING SPARK")
    assert len(load_metadata("Wiley"))
    print("DONE RUNNING!! ")