import getpass
import tempfile
import traceback

from common import ANNOTATION_SCHEME_VERSION, get_annotation_local_path

import worker_funcs
import worker_models
from itertools import groupby
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils
from databricks_cli.sdk.api_client import ApiClient
import databricks_cli.dbfs.api
from databricks_cli.dbfs.api import DbfsApi
from databricks_cli.dbfs.dbfs_path import DbfsPath
import datetime
import json

databricks_cli.dbfs.api.error_and_quit = st.error
databricks_cli.dbfs.api.click.echo = st.info

MODELS_BATCH_SIZE = 20
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
        "mllake_lct": f"{SNAPSHOT_DIR}/00Dd0000000eekuEAA-LiveChatTranscript-sp_9c412e88-3ce2-4a07-8c9b-f567c625967a/data/ReceivedAt_day=2022-02-02",
        "mllake_case": f"{SNAPSHOT_DIR}/00Dd0000000eekuEAA-Case-sp_2f8d4c3a-9d97-43f2-98ce-d837f5d44750/data/ReceivedAt_day=2022-02-02",
    }
}


def get_tenant_temp_dir(tenant_name):
    return f"{DATABRICKS_S3_DIR}/{_get_lct_temp_dir(tenant_name)}"


def get_tenant_temp_data_dir(tenant_name):
    return f"{get_tenant_temp_dir(tenant_name)}/lct_with_case_json"


def get_tenant_batch_temp_path(tenant_name, batch_name):
    return f"{get_tenant_temp_dir(tenant_name)}/{batch_name}.json"


@st.cache
def _get_lct_temp_dir(tenant_name):
    tenant_doc = TENANTS[tenant_name]
    curdate = datetime.datetime.utcnow().strftime("%Y%m%d")
    for temp_dir in tenant_doc["temp_dir"]:
        expiry = temp_dir.split("_")[-1]
        if expiry > curdate:
            print(f"Using temp dir {temp_dir}")
            return temp_dir

    raise FileNotFoundError()


@st.cache(allow_output_mutation=True)
def _get_spark():
    getpass.getuser()
    spark = SparkSession.getActiveSession()
    if not spark:
        spark = SparkSession.builder.appName(
            f"Annotation-Tool-{getpass.getuser()}"
        ).getOrCreate()
        spark.sparkContext.addPyFile(worker_funcs.__file__)
        spark.sparkContext.addPyFile(worker_models.__file__)

    return spark


def _get_spark_context():
    return _get_spark().sparkContext


@st.cache(allow_output_mutation=True)
def _get_dbutils():
    return DBUtils(_get_spark())


def _parallelize(lst, n_partitions=None):
    assert len(lst) > 0
    return _get_spark_context().parallelize(lst, n_partitions or min(16, len(lst)))


def _file_key(dct):
    return dct["chat_file"]


def load_chats(chat_metadata):
    chats_by_file = [
        (file, [md["uid"] for md in mds])
        for file, mds in groupby(sorted(chat_metadata, key=_file_key), key=_file_key)
    ]

    chats = _parallelize(chats_by_file).flatMap(worker_funcs.read_chats).collect()

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


def _process_raw_data(tenant_name):
    chat_temp_dir = get_tenant_temp_data_dir(tenant_name)
    raw_lct_files = _get_dbutils().fs.ls(TENANTS[tenant_name]["mllake_lct"])
    lct_rdd_by_case_id = (
        _parallelize([f.path for f in raw_lct_files])
        .flatMap(worker_funcs.read_raw_lcts)
        .keyBy(lambda record: record["case_id"])
    )

    raw_case_files = _get_dbutils().fs.ls(TENANTS[tenant_name]["mllake_case"])
    case_rdd_by_id = (
        _parallelize([f.path for f in raw_case_files])
        .flatMap(worker_funcs.read_raw_cases)
        .keyBy(lambda record: record["Id"])
    )

    lct_rdd_by_case_id.join(case_rdd_by_id).reduceByKey(
        lambda v1, v2: v1 # drop duplicates :(
    ).map(
        lambda id_to_lct_and_chat: dict(
            id_to_lct_and_chat[1][0], case=id_to_lct_and_chat[1][1]
        )
    ).map(
        lambda obj: json.dumps(obj, default=str)
    ).saveAsTextFile(chat_temp_dir)


def _get_chat_files(tenant_name):
    chat_temp_dir = get_tenant_temp_data_dir(tenant_name)
    chat_files = _safe_ls(chat_temp_dir)
    if chat_files is None:
        _process_raw_data(tenant_name)
        chat_files = _safe_ls(chat_temp_dir)
    return chat_files


def load_metadata(tenant_name):
    chat_files = _get_chat_files(tenant_name)
    return (
        _parallelize([f.path for f in chat_files if "part-" in f.path])
        .flatMap(worker_funcs.read_chats_metadata)
        .collect()
    )


def fetch_batch_meta(tenant_name, batch_name):
    batch_path = _get_batch_path(tenant_name, batch_name)
    api_client = _get_dbfs_api_client()
    with tempfile.TemporaryDirectory() as td:
        tfpath = f"{td}/{batch_name}"
        api_client.get_file(batch_path, tfpath, True)
        with open(tfpath, "r") as tf:
            return json.load(tf)


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


def _get_batches_path(tenant_name):
    return DbfsPath(f"{DBFS_SHARED_DIR}/{tenant_name}/batches", False)


def _get_batch_path(tenant_name, batch_name):
    return DbfsPath(
        f"{DBFS_SHARED_DIR}/{tenant_name}/batches/{batch_name}/batch.json", False
    )


def get_annotation_path(tenant_name, batch_name, annotation_type):
    username = getpass.getuser()
    return DbfsPath(
        f"{DBFS_SHARED_DIR}/{tenant_name}/batches/{batch_name}/annotations/"
        f"{annotation_type}/"
        f"{ANNOTATION_SCHEME_VERSION}/{username}.anno",
        False,
    )


def load_tenants():
    return list(TENANTS.keys())


@st.cache
def load_batch_names(tenant_name):
    base_path = _get_batches_path(tenant_name)
    api_client = _get_dbfs_api_client()

    if not api_client.file_exists(base_path):
        api_client.mkdirs(base_path)
    batchs = api_client.list_files(base_path)
    return [b.dbfs_path.basename for b in batchs]


def create_batch(tenant_name, batch_name, batch_size, turn_range):
    chat_files = _get_chat_files(tenant_name)
    chat_meta_rdd = (
        _parallelize([f.path for f in chat_files if "part-" in f.path])
        .flatMap(worker_funcs.read_chats_metadata)
        .filter(
            lambda doc: doc["n_turns"] > turn_range[0]
            and doc["n_turns"] < turn_range[1]
        )
    )
    n = chat_meta_rdd.count()
    meta_data = chat_meta_rdd.sample(False, float(batch_size * 2.0 / n)).collect()[
        :batch_size
    ]

    with tempfile.TemporaryDirectory() as tdir:
        tfname = f"{tdir}/{batch_name}"
        metadata_for_batch = {
            "name": batch_name,
            "size": batch_size,
            "create_date": datetime.datetime.now().strftime("%d-%m-%Y"),
            "created_by": getpass.getuser(),
            "chat_uids": [doc["uid"] for doc in meta_data],
        }
        with open(tfname, "w") as tf:
            json.dump(metadata_for_batch, tf)

        _get_dbfs_api_client().put_file(
            tfname, _get_batch_path(tenant_name, batch_name), True
        )


def upload_annotation(tenant_name, batch_name, annotation_type, annotation):
    local_file = get_annotation_local_path(tenant_name, batch_name, annotation_type)
    with open(local_file, "w") as tfile:
        json.dump(annotation, tfile)
    _get_dbfs_api_client().put_file(
        local_file, get_annotation_path(tenant_name, batch_name, annotation_type), True
    )


def fetch_annotation(tenant_name, batch_name, annotation_type):
    try:
        anoo_path = get_annotation_path(tenant_name, batch_name, annotation_type)
        local_file = get_annotation_local_path(tenant_name, batch_name, annotation_type)
        _get_dbfs_api_client().get_file(anoo_path, local_file, True)
        with open(local_file, "r") as tf:
            return json.load(tf)
    except:
        return {}


def _assert_models_are_available(temp_s3_bucket):
    for model in worker_models.models.values():
        dbfs_path = model.model_name_or_path
        s3_path = worker_models._infer_model_path(dbfs_path, temp_s3_bucket)
        print(f"{dbfs_path} -> {s3_path}")
        if dbfs_path == s3_path:
            continue

        target_files = _safe_ls(s3_path)
        if not target_files:
            return st.error(
                f"Model not available on temp s3 bucket. Please run on  Databricks notebook: "
                f"dbutils.fs.cp('{dbfs_path}', '{s3_path}', recurse=True)")


def apply_models(tenant_name, batch_name, batch, model_names):
    chat_uids_and_texts = [
        (chat["uid"], chat["chat_text"])
        for chat in batch["chats"]
    ]
    n_batches = len(chat_uids_and_texts) // MODELS_BATCH_SIZE
    temp_s3_bucket = get_tenant_temp_dir(tenant_name)
    _assert_models_are_available(temp_s3_bucket)
    all_model_preds = _parallelize(model_names, len(model_names)).cartesian(
        _parallelize(chat_uids_and_texts, n_batches)
    ).mapPartitions(
        lambda partition: worker_models.run_model_on_partition(partition, temp_s3_bucket)
    ).collect()

    chat_by_uid = {chat["uid"]: chat for chat in batch["chats"]}
    for model_name, model_preds in all_model_preds:
        for uid, pred in model_preds.items():
            chat_by_uid[uid].setdefault("preds", {})[model_name] = pred

    save_batch(tenant_name, batch_name, batch)
    return batch


def get_saved_batch(tenant_name, batch_name):
    try:
        text = _get_dbutils().fs.head(
            get_tenant_batch_temp_path(tenant_name, batch_name),
            2147483647
        )
        return json.loads(text)
    except Exception as e:
        if "java.io.FileNotFoundException" not in str(e):
            print(traceback.format_exc())

        return None


def save_batch(tenant_name, batch_name, batch):
    _get_dbutils().fs.put(
        get_tenant_batch_temp_path(tenant_name, batch_name),
        json.dumps(batch),
        True
    )
