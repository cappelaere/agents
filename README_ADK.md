# Installing Python 3.11 and using the ADK
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

Plan B:
Get API key from instance credential

Update WO_INSTANCE and WO_API_KEY and update the env file
and install orchestrate stuff

```
source orchestrate311/bin/activate

orchestrate env add -n saas-wxo -u $WO_INSTANCE
orchestrate env activate saas-wxo --api-key $WO_API_KEY 
orchestrate env list

orchestrate server start -l -e env

INFO] - Auto-detecting local IP address for async tool callbacks...
[INFO] - Auto-configured CALLBACK_HOST_URL to: http://172.18.0.1:4321
[INFO] - Logging into Docker registry: registry.us-south.watson-orchestrate.cloud.ibm.com ...
[ERROR] - Error logging into Docker:
Error response from daemon: Get "https://registry.us-south.watson-orchestrate.cloud.ibm.com/v2/": received unexpected HTTP status: 500 Internal Server Error

``` 
# Then go to https://localhost:8765 in your browser.
