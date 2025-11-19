# Installing Python 3.11 and using the ADK
```
sudo dnf install -y python3.11 python3.11-devel
sudo dnf install -y net-tools

python3.11 -m venv venv
source venv/bin/activate
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
source venv/bin/activate

orchestrate env add -n saas-wxo -u $WO_INSTANCE
orchestrate env activate saas-wxo --api-key $WO_API_KEY 
orchestrate env list

orchestrate server start -l -e env

