import os
ANNOTATION_SCHEME_VERSION = "v1"
DATA_DIR = "data"

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

def get_annotation_local_path(tenant_name, batch_name):
    return f"{DATA_DIR}/{tenant_name}_{batch_name}_{ANNOTATION_SCHEME_VERSION}.anno"
