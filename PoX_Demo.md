# PoX Demo

## Meet you customized agents
### Naviguate to IBM Cloud Orchestrate SaaS itz-watsonx-24
[launch Orchestrate Instance](https://cloud.ibm.com/services/watsonx-orchestrate/

Resource list/AI Machine Learning/
Select WatsonX Orchestrate
Launch watsonx Orchestrate
Manage Agents
Describe all the customized agents created for this PoX
    - Knowledge Agent
    - METOC Agent
    - Sea Ice Agent
    - AIS Agent

### Knowledge Agent
Give me a Sea Ice Index anomaly explanation?
What are the satellite sensors for sea-ice monitoring?
What is the NOAA–USCG Arctic alignment?
Real-time systems for ice/weather/vessel fusion.

### METOC Agent
What is the 24-hour wave height forecast near Barrow?
Give me the 5-day marine forecast for Prudhoe Bay
Find coordinates for Utqiaġvik, AK
Twhat is tmorrow’s temperature forecast there?
What is the 48-hour wind forecast for 71.29, –156.79?

### Sea Ice agent
Show 11/30/2025  sea-ice concentration near Latitude: 85° N and Longitude: 156.7886 ° W.
Mean/min/max ice concentration in bbox [70,-120,85,-125] as of 11/30/2025

### AIS Agent
Show 20 ships near 50nm from Juneau, AK
generate geojson object
Show vessel info MMSI=
Show photo

## Create Agent in the console
Create knowledge agent with NORTHCOM pdf

**Mission & Roles**

What is the primary mission of NORAD and USNORTHCOM?
How do NORAD and NORTHCOM divide responsibilities for homeland defense?
What is the role of NORAD in aerospace warning and control?

**Strategic Priorities**

What are the top priorities outlined in the 2025 posture statement?
How does the NORAD-USNORTHCOM strategy address global competition?
What are the key objectives in the Strategic Guidance document?

**Threats & Challenges**

What emerging threats are identified in the latest posture statement?
How does NORTHCOM plan to counter hypersonic missile threats?
What cyber defense measures are mentioned in the strategy summary?

**Operations & Exercises**

What major exercises does NORAD participate in annually?
Describe the role of Operation Noble Eagle.
How does NORTHCOM coordinate with FEMA during disaster response?

**Partnerships & Coordination**

Which countries are NORAD partners?
How does NORTHCOM collaborate with Canada and Mexico?
What interagency partnerships are highlighted in the posture statement?

**Capabilities & Modernization**

What modernization efforts are underway for NORAD radar systems?
Explain the concept of “All-Domain Awareness” mentioned in the strategy.
What investments are planned for Arctic defense?

**Historical & Organizational**

When was NORAD established and why?
Who is the current commander of NORAD and NORTHCOM?
How has the mission evolved since 9/11?

## Creating Agents with ADK
Show VSC, DEMO.md
source dev-shell.sh
orchestrate env activate $ENV_NAME --api-key $WO_API_KEY
orchestrate models list
AI Agent Studio [AIOPs](https://www.ibm.com/watsonx/developer/agents/quickstart/)

## Hybrid and Distributed Agent Deployment
We can show our VM on Techzone
https://cloud.ibm.com/infrastructure/compute/vs/us-east-2~0767_2a3e2166-8f92-44f6-a6c5-169642b36577/overview
rhel-pgc-3 on itz-watsonx-24

docker ps
docker logs ais_agent --follow

## Agent Discovery and Reuse
Left menu bar / Discover

## Agent Analytics & Tracing

Langfuse:
https://us.cloud.langfuse.com/project/cmi7n2hsq0182ad07ey26cq0d/traces?peek=14fd8f69aa543ede9ea3e717dbd265f5&timestamp=2025-11-30T22%3A52%3A14.540Z

## Orchestration
Arctic Region Assistant
AIS + Map + METOC + SEAICE...

https://geojson.io/#map=3.63/68.62/-138.27

## Autonomous Workflows

Daily Brief report:
http://150.240.3.116:8120/daily_brief

Vessel Risk Report
MMSI: 345876980
http://150.240.3.116:8120/vessel_risk_report

http://150.240.3.116:8120/
http://150.240.3.116:8120/map

