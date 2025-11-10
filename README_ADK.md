# Installing and using the ADK
```
sudo dnf install -y python3.11 python3.11-devel
sudo dnf install -y net-tools

python3.11 -m venv orchestrate311
source orchestrate311/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install ibm-watsonx-orchestrate
```

[https://developer.watson-orchestrate.ibm.com/getting_started/installing#ibm-cloud]

Go to orchestrate, click on user profile icon/settings
Get Service instance Url and Generate API Key
! Create button is disabled


Update WO_INSTANCE and WO_API_KEY and update the env file
and install orchestrate stuff

```
orchestrate env add -n remote_env -u $WO_INSTANCE
orchestrate env activate remote_env --api-key $WO_API_KEY --type ibm_iam
orchestrate env list

export ORCH_CALLBACK_URL="http://$HOST:4321"

orchestrate server start -l -e env
``` 