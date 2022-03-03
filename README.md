# Summarization Annotation Tool

An Streamlit-based annotation tool.
To use, 
1. Create and activate a virtualenv (e.g. `virtualenv -p <PYTHON3.8> .env`)
2. Install the requirements: `pip install -r requirements.txt`
3. Create a Databricks Access Token from https://databricks.prod.platform.einstein.com/#setting/account
4. Configure Databricks Connect by running `databricks-connect configure`, or create a config 
   file `.databricks-connect` in your home dir:
```json
{
  "host": "https://databricks.prod.platform.einstein.com/",
  "token": "<YOUR_TOKEN_HERE>",
  "cluster_id": "<SPARK_CLUSTER>",
  "org_id": "0",
  "port": "15001"
}
```
We're currently using Erez's spark cluster which has Wiley data access, "0216-080021-basin0000660" 

Run the app and visit http://localhost:8501

```shell
streamlit run annotation.py
```

VPN is needed to access Databricks. The cluster will be started if it's not already running, and a 
Spark job will be run on the driver, launching workers to complete it, all of which might take 
a few minutes.    

