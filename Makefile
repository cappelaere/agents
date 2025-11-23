.PHONY: build deploy down

build:
	docker compose build metoc_agent
	docker compose build seaice_agent
	docker compose build ais_agent
	docker compose build ports_agent
	docker compose build map_agent
	docker compose build mcp_ais
	docker compose build mcp_metoc

deploy:
	docker compose up --no-deps -d metoc_agent
	docker compose up --no-deps -d seaice_agent
	docker compose up --no-deps -d ais_agent
	docker compose up --no-deps -d ports_agent
	docker compose up --no-deps -d map_agent
	docker compose up --no-deps -d mcp_ais
	docker compose up --no-deps -d mcp_metoc

down:
	docker compose down metoc_agent
	docker compose down seaice_agent
	docker compose down ais_agent
	docker compose down ports_agent
	docker compose down map_agent
	docker compose down mcp_ais
	docker compose down mcp_metoc