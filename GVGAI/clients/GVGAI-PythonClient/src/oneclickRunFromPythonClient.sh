#!/bin/bash

# Games:
# 11 -> BoulderDash
# 43 -> IceAndFire
# 16 -> Catapults

game_id=43
server_dir=../../..
agent_name=MyAgent.ExampleAgent
sh_dir=utils


DIRECTORY='./logs'
if [ ! -d "$DIRECTORY" ]; then
  mkdir ${DIRECTORY}
fi

# Run the client with visualisation on
#python3 TestLearningClient.py -gameId ${game_id} -agentName ${agent_name} -serverDir ${server_dir} -shDir ${sh_dir}
# Run the client with visualisation
python3 TestLearningClient.py -gameId ${game_id} -agentName ${agent_name} -serverDir ${server_dir} -shDir ${sh_dir} -visuals
