import databricks_funcs
from itertools import groupby
import streamlit as st
from pyspark.sql import SparkSession

DATABRICKS_S3_DIR   = "s3://usw2-sfdc-ecp-prod-databricks-users"
DATABRICKS_TEMP_DIR = f"{DATABRICKS_S3_DIR}/databricks_2051_ai-prod-00Dd0000000eekuEAA/20220202_20220304"

def _get_spark():
    spark = SparkSession.getActiveSession()
    if not spark:
        spark = SparkSession.builder.getOrCreate()
        spark.sparkContext.addPyFile(databricks_funcs.__file__)

    return spark

def _get_spark_context():
    return _get_spark().sparkContext

def _get_dbutils():
    from pyspark.dbutils import DBUtils
    return DBUtils(_get_spark())

def _file_key(dct):
    return dct['chat_file']

def load_chats(chat_metadata):
    chats_by_file = [
        (file, [md['uid'] for md in mds])
        for file, mds in groupby(sorted(chat_metadata, key=_file_key), key=_file_key)
    ]

    chats = _get_spark_context().parallelize(
        chats_by_file,
        len(chats_by_file)
    ).flatMap(
        databricks_funcs.read_chats
    ).collect()

    if len(chats) < len(chat_metadata):
        st.warning(f"Retrieved {len(chats)}/{len(chat_metadata)} chats")

    chats_by_uid = {
        chat['uid']: chat
        for chat in chats
    }
    return [
        dict(chat_md, **chats_by_uid[chat_md["uid"]])
        for chat_md in chat_metadata
        if chat_md["uid"] in chats_by_uid
    ]

def load_metadata():
    chat_files = [
        f.path for f in _get_dbutils().fs.ls(f"{DATABRICKS_TEMP_DIR}/chats.full.json")
        if 'part-' in f.path
    ]
    return _get_spark_context().parallelize(
        chat_files, len(chat_files)
    ).flatMap(databricks_funcs.read_chats_metadata).collect()
