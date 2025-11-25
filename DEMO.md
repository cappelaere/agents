# Arctic Region DSS PoX

# Visual Studio Code window connected remoteley to TechZone Instance

# Go to Techzone to show Virtual Server on itz-watsonx-24
[rhel-lgc-3](https://cloud.ibm.com/infrastructure/compute/vs/us-east-2~0767_2a3e2166-8f92-44f6-a6c5-169642b36577/overview)

# Naviguate to IBM Cloud Orchestrate SaaS itz-watsonx-24
[launch Orchestrate Instance](https://cloud.ibm.com/services/watsonx-orchestrate/crn%3Av1%3Abluemix%3Apublic%3Awatsonx-orchestrate%3Aus-south%3Aa%2Fa8b6f41c856d4312a3bc7a99e5eab392%3A011eb689-9996-483f-96e3-89410b775a5d%3A%3A?paneId=manage)

# Create Knowledge Agent

# Create METOC AGENT (MCP Server)
http://150.240.3.116:8201/mcp


# Traceability LangFuse


# Docs
[Doc](https://www.ibm.com/docs/en/watsonx/watson-orchestrate/base)

[ADK](https://developer.watson-orchestrate.ibm.com)

## PreReq
Here is the Orchestrate Instance: https://api.us-south.watson-orchestrate.cloud.ibm.com/instances/011eb689-9996-483f-96e3-89410b775a5d

## Activate env
orchestrate env activate $ENV_NAME --api-key $WO_API_KEY
orchestrate env list

APP_ID=mcp_metoc
MCP_HOST="http://150.240.3.116:8201/mcp"
MCP_TOOLKIT=mcp_metoc_toolkit

## Create a connection
orchestrate connections add -a $APP_ID

## Remove connection
orchestrate connections remove --app-id $APP_ID

## Create toolkit as MCP Server
orchestrate toolkits import --kind mcp --name $MCP_TOOLKIT --description "Generates atmospheric and marine weather forcast and provide geocoding support"  --url $MCP_HOST --transport streamable_http --tools  "*" --app-id $APP_ID


orchestrate toolkits import -f yml/mcp_metoc_toolkit.yml -a $APP_ID

## List toolkits
orchestrate toolkits list -v

## Remove toolkit
orchestrate toolkits remove -n $MCP_TOOLKIT

## Create agent using yml file
orchestrate agents import -f yaml/metoc_agent.yml

## Manage agent
```
orchestrate agents list -v
orchestrate agents remove --name metoc_agent --kind native
```

## Manage models
```
orchestrate models list
```

## Clean up
```
APP_ID=mcp_metoc
MCP_TOOLKIT=mcp_metoc_toolkit
MCP_HOST=http://150.240.3.116:8201/mcp
orchestrate agents remove --name metoc_agent --kind native
orchestrate toolkits remove -n $MCP_TOOLKIT
orchestrate connections remove --app-id $APP_ID

APP_ID=mcp_ais
MCP_TOOLKIT=mcp_ais_toolkit
MCP_HOST=http://150.240.3.116:8200/mcp
orchestrate agents remove --name metoc_agent --kind native
orchestrate toolkits remove -n $MCP_TOOLKIT
orchestrate connections remove --app-id $APP_ID

```
## Restart
```
APP_ID=mcp_metoc
MCP_TOOLKIT=mcp_metoc_toolkit
MCP_HOST=http://150.240.3.116:8201/mcp
orchestrate connections add -a $APP_ID
orchestrate connections configure -a $APP_ID --env draft --type team --kind key_value
orchestrate connections set-credentials -a $APP_ID --env draft -e "SECURE_ENVIRONMENT_VARIABLE=0"

# This does not work.  No such option: -f
# orchestrate toolkits import -f yml/mcp_metoc_toolkit.yml -a $APP_ID

# Doing this instead
orchestrate toolkits import --kind mcp --name $MCP_TOOLKIT --description "Generates atmpospheric and marine weather forecast and provide geocoding support"  --url $MCP_HOST --transport streamable_http --tools  "*" --app-id $APP_ID

orchestrate agents import -f yaml/metoc_agent.yml

# ----------------------------------------------------------
# Loading AIS agent from MCP AIS server
APP_ID=mcp_ais
MCP_TOOLKIT=mcp_ais_toolkit
MCP_HOST=http://150.240.3.116:8200/mcp

orchestrate connections add -a $APP_ID
orchestrate connections configure -a $APP_ID --env draft --type team --kind key_value
orchestrate connections set-credentials -a $APP_ID --env draft -e "SECURE_ENVIRONMENT_VARIABLE=0"

orchestrate toolkits import --kind mcp --name $MCP_TOOLKIT --description "tools for ship tracking"  --url $MCP_HOST --transport streamable_http --tools  "*" --app-id $APP_ID

orchestrate agents import -f yaml/ais_agent.yml

```

AI Agent Studio [AIOPs](https://www.ibm.com/watsonx/developer/agents/quickstart/)
