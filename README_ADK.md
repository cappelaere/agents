# Orchestrate ADK
**ADK Reference:** [Docs](https://developer.watson-orchestrate.ibm.com/getting_started/installing#ibm-cloud)

## Prereqs
Go to propoer cloud instance: $CLOUD_INSTANCE
Go to [IBM Cloud, Orchestrate](https://cloud.ibm.com/services/watsonx-orchestrate/crn%3Av1%3Abluemix%3Apublic%3Awatsonx-orchestrate%3Aus-south%3Aa%2Fa8b6f41c856d4312a3bc7a99e5eab392%3A011eb689-9996-483f-96e3-89410b775a5d%3A%3A?paneId=manage)

Get the service url and API key.
Update .env.sh

## Python 3.11 and dev tools
```
sudo dnf install -y python3.11 python3.11-devel
sudo dnf install -y net-tools
python3.11 -m venv venv
source venv/bin/activate
```

## ADK Installation or upgrade
```
python -m pip install --upgrade pip setuptools wheel
pip install  --upgrade ibm-watsonx-orchestrate

```
## Configure Environment on IBM Cloud
Go to Watson Orchestrate and launch it
Go to user icon / settings
Generate an API_KEY and update .env.sh

```    
orchestrate env add -n $ENV_NAME --url $WO_INSTANCE --type ibm_iam
orchestrate env activate $ENV_NAME --api-key $WO_API_KEY

```
## List environments
```
orchestrate env list
```

## Create .orchestrate_env file

## Start server
```
orchestrate server start -e .orchestrate_env --with-ibm-telemetry

orchestrate chat start
```
Once installed, the following services become available:
[OpenAPI Docs](http://localhost:4321/docs)

[API Base URL:](http://localhost:4321/api/v1)

 This API documentation is also available in the watsonx Orchestrate Developer Edition APIs.

**DOES NOT WORK**

Go to [Observabiloty Dashboard](https://localhost:8765)

**END**

## Orchestrate Copilot 
### Install VSC Plugin
[Install](https://developer.watson-orchestrate.ibm.com/copilot/installing_copilot) "watsonx Orchestrate ADK" plugin

