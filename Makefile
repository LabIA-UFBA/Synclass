#
# autor: Marcos Vinícius
# email: ferreiraviniciussdi@gmail.com
#
.PHONY : help

#---------- # 
# VARIABLEs #
#---------- #
# 
include settings.env

#--------- # 
# ACTIONS  #
#--------- #

build:
	@echo "Build image $(IMAGE)" 
	@docker-compose --env-file settings.env build

run:
	@docker-compose --env-file settings.env up -d

down:
	@docker-compose --env-file settings.env down

exec:
	@docker exec -it $(CONTAINER) bash

clean:
	@echo "remove ans stop all containers"
	@docker stop $(shell docker ps -aq)
	@docker rm $(shell docker ps -aq)

down-f1:
	@echo "remove ans stop all containers"
	@docker stop $(shell docker ps -aq)
	@docker rm $(shell docker ps -aq)

vlogs:
	@docker logs $(CONTAINER) -f


trun: down run vlogs
