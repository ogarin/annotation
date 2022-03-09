import tempfile

import databricks_funcs
from itertools import groupby
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils
from databricks_cli.sdk.api_client import ApiClient
from databricks_cli.dbfs.cli import DbfsApi
from databricks_cli.dbfs.dbfs_path import DbfsPath
import datetime
import json

DBFS_SHARED_DIR = "dbfs:/FileStore/tables/summarization"
DATABRICKS_S3_DIR = "s3://usw2-sfdc-ecp-prod-databricks-users"
SNAPSHOT_DIR = "s3://prod-usw2-datasets/snapshots"

TENANTS = {
    "Wiley": {
        "temp_dir": [
            "databricks_2051_ai-prod-00Dd0000000eekuEAA/20220202_20220304",
            "databricks_2051_ai-prod-00Dd0000000eekuEAA/20220304_20220403",
            "databricks_2051_ai-prod-00Dd0000000eekuEAA/20220403_20220503",
            "databricks_2051_ai-prod-00Dd0000000eekuEAA/20220503_20220602",
            "databricks_2051_ai-prod-00Dd0000000eekuEAA/20220602_20220702",
            "databricks_2051_ai-prod-00Dd0000000eekuEAA/20220702_20220801",
            "databricks_2051_ai-prod-00Dd0000000eekuEAA/20220801_20220831",
            "databricks_2051_ai-prod-00Dd0000000eekuEAA/20220831_20220930",
            "databricks_2051_ai-prod-00Dd0000000eekuEAA/20220930_20221030",
            "databricks_2051_ai-prod-00Dd0000000eekuEAA/20221030_20221129",
        ],
        "mllake_lct": f"{SNAPSHOT_DIR}/00Dd0000000eekuEAA-LiveChatTranscript-sp_9c412e88-3ce2-4a07-8c9b-f567c625967a/data/ReceivedAt_day=2022-02-02/",
    }
}


def get_tenant_temp_dir(tenant_name):
    return f"{DATABRICKS_S3_DIR}/{_get_lct_temp_dir(tenant_name)}/lct_json"


def _get_lct_temp_dir(tenant_name):
    tenant_doc = TENANTS[tenant_name]
    curdate = datetime.datetime.utcnow().strftime("%Y%m%d")
    for temp_dir in tenant_doc["temp_dir"]:
        expiry = temp_dir.split("_")[-1]
        if expiry > curdate:
            print(f"Using temp dir {temp_dir}")
            return temp_dir

    raise FileNotFoundError()


def _get_spark():
    spark = SparkSession.getActiveSession()
    if not spark:
        spark = SparkSession.builder.getOrCreate()
        spark.sparkContext.addPyFile(databricks_funcs.__file__)

    return spark


def _get_spark_context():
    return _get_spark().sparkContext


def _get_dbutils():
    return DBUtils(_get_spark())


def _parallelize(lst):
    assert len(lst) > 0
    return _get_spark_context().parallelize(lst, min(16, len(lst)))


def _file_key(dct):
    return dct["chat_file"]


def load_chats(chat_metadata):
    chats_by_file = [
        (file, [md["uid"] for md in mds])
        for file, mds in groupby(sorted(chat_metadata, key=_file_key), key=_file_key)
    ]

    chats = _parallelize(chats_by_file).flatMap(databricks_funcs.read_chats).collect()

    if len(chats) < len(chat_metadata):
        st.warning(f"Retrieved {len(chats)}/{len(chat_metadata)} chats")

    chats_by_uid = {chat["uid"]: chat for chat in chats}
    return [
        dict(chat_md, **chats_by_uid[chat_md["uid"]])
        for chat_md in chat_metadata
        if chat_md["uid"] in chats_by_uid
    ]


def _safe_ls(path):
    try:
        return _get_dbutils().fs.ls(path)
    except Exception as e:
        if "java.io.FileNotFoundException" in str(e):
            return None
        else:
            raise


def _process_raw_lct(tenant_name):
    chat_temp_dir = get_tenant_temp_dir(tenant_name)
    mllake_lct = TENANTS[tenant_name]["mllake_lct"]
    raw_lct_files = _get_dbutils().fs.ls(mllake_lct)
    _parallelize([f.path for f in raw_lct_files]).flatMap(
        databricks_funcs.read_raw_lct
    ).map(json.dumps).saveAsTextFile(chat_temp_dir)


def load_metadata(tenant_name):
    chat_temp_dir = get_tenant_temp_dir(tenant_name)
    chat_files = _safe_ls(chat_temp_dir)
    if chat_files is None:
        _process_raw_lct(tenant_name)
        chat_files = _safe_ls(chat_temp_dir)

    return (
        _parallelize([f.path for f in chat_files if "part-" in f.path])
        .flatMap(databricks_funcs.read_chats_metadata)
        .collect()
    )


def _get_dbfs_api_client():
    spark = _get_spark()
    api_client = ApiClient(
        host=spark.conf.get("spark.databricks.service.address"),
        token=spark.conf.get("spark.databricks.service.token"),
    )
    return DbfsApi(api_client)


def upload_file_to_shared_dir(file, subdir="", overwrite=False):
    _get_dbfs_api_client().put_file(
        file, DbfsPath(f"{DBFS_SHARED_DIR}/{subdir}", False), overwrite
    )


def _get_batch_path(tenant_name, suffix=""):
    return DbfsPath(f"{DBFS_SHARED_DIR}/{tenant_name}/batchs/{suffix}", False)


def load_tenants():
    return list(TENANTS.keys())


def load_batch(tenant_name):
    base_path = _get_batch_path(tenant_name)
    api_client = _get_dbfs_api_client()

    if not api_client.file_exists(base_path):
        api_client.mkdirs(base_path)
    batchs = api_client.list_files(base_path)
    return batchs


def create_batch(tenant_name, batch_name, batch_size, turn_range):
    chat_temp_dir = get_tenant_temp_dir(tenant_name)
    chat_files = _safe_ls(chat_temp_dir)
    chat_meta_rdd = (
        _parallelize([f.path for f in chat_files if "part-" in f.path])
        .flatMap(databricks_funcs.read_chats_metadata)
        .filter(
            lambda doc: doc["n_turn"] > turn_range[0] and doc["n_turn"] < turn_range[1]
        )
    )
    n = chat_meta_rdd.count()
    meta_data = chat_meta_rdd.sample(batch_size * 2 / n).collect()

    with tempfile.NamedTemporaryFile() as tf:
        for meta_doc in meta_data:
            tf.write(json.dumps(meta_doc))
            tf.write("\n")
        tf.close()
        _get_dbfs_api_client().put_file(
            tf.name, _get_batch_path(tenant_name, batch_name)
        )
