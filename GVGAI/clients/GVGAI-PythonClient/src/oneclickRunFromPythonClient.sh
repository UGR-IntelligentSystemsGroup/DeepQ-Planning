#!/bin/bash

game_id=11
server_dir=../../..
agent_name=MyAgent.Agent
sh_dir=utils


DIRECTORY='./logs'
if [ ! -d "$DIRECTORY" ]; then
  mkdir ${DIRECTORY}
fi

# Run the client with visualisation on
python TestLearningClient.py -gameId ${game_id} -agentName ${agent_name} -serverDir ${server_dir} -shDir ${sh_dir} -visuals
# Run the client with visualisation off
#python TestLearningClient.py -gameId ${game_id} -agentName ${agent_name} -serverDir ${server_dir} -shDir ${sh_dir}
