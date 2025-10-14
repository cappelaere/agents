AGENTS := metoc_agent ais_agent seaice_agent
.PHONY: build $(AGENTS:%=build-%) clean

metoc_agent:
	docker build -f Dockerfile_metoc_agent -t metoc_agent:latest .

ais_agent:	
	docker build -f Dockerfile_ais_agent -t ais_agent:latest .

seaice_agent:
	docker build -f Dockerfile_seaice_agent -t seaice_agent:latest .

clean:
	@for a in $(AGENTS); do \
	  docker rmi $$a:latest || true; \
	done