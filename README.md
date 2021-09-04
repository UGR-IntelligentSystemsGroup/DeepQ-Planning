# PlanningWithSubgoals

This project implements a planning-based agent which learns to select goals using Deep Q-Learning. We have applied it to play videogames included in the [GVGAI][http://www.gvgai.net/] framework. For more information about the project, please refer to the following [Workshop Publication][https://drive.google.com/file/d/1GV-fPLIFe1nHtM56OA3SPpAfENkmjOKv/view]. A paper with a more recent version of our work has been accepted for the [CAEPIA][https://caepia20-21.uma.es/] Conference and will be linked here once published.

This project has been partially funded by the Spanish MICINN R\&D Project Ref. RTI2018-098460-B-I00.

## Requirements

Here we detail our code requirements. Our code has been implemented in Python using the [Anaconda][https://www.anaconda.com/] 4.7.12 distribution, on a local machine running the [OpenJDK][https://openjdk.java.net/] version 11.0.6 and Ubuntu 18.04.3 x86_64 as the OS. [TensorFlow][https://www.tensorflow.org/] 1.10 has been our library of choice to implement the Machine Learning models, such as Deep Q-Learning. For installing the correct Python version, libraries and dependencies, we recommend installing Anaconda and then importing the following [conda environment](../../blob/DQP-allgames/env_setup/environment.yml).

## Code Structure

Here we explain the most important files of our code and their location in the directory tree of the project. Firstly, the **DQP-allgames** branch corresponds with the most recent branch of the project. Here, we have implemented an agent with combines Deep Q-Learning and Planning to play three different GVGAI games: BoulderDash, IceAndFire and Catapults. The most important files in this branch are listed below:

⋅⋅* [**Agent.py.**](../../blob/DQP-allgames/GVGAI/clients/GVGAI-PythonClient/src/MyAgent/Agent.py) This is the most important file of the project. It implements the agent which plays the game. The *init* method (not to be mistaken with *__init__*) is called at the start of every level. Then, in each turn, the *act* method is called. It receives the current state of the game as a parameter and needs to return the next action to be executed by the agent. Finally, the *result* method is called once the level has finished, due to the agent winning the game or losing it. 

..* [**oneclickRunFromPythonClient.sh.**](../../blob/DQP-allgames/GVGAI/clients/GVGAI-PythonClient/src/oneclickRunFromPythonClient.sh) When called, this file starts up the server and runs the corresponding GVGAI game (with the agent playing it). It determines the game to be played and if the visual interface should be shown or not, among other things.

..* [**LearningModel.py.**](../../blob/DQP-allgames/GVGAI/clients/GVGAI-PythonClient/src/oneclickRunFromPythonClient.sh) This file implements the TensorFlow code of the Deep Q-Learning model which selects goals. The architecture used corresponds to a Convolutional Neural Network which receives the current state of the game along with a possible goal as an image-like encoding.

..* [**ejecutar_create_dataset.py.**](../../blob/DQP-allgames/GVGAI/clients/GVGAI-PythonClient/src/ejecutar_create_dataset.py) This file is in charge of automatically creating the datasets used to train the agent to select goals. It plays every game level in sequence, selecting and achieving goal at random and collecting the experience needed to populate the training datasets. To execute it, open up a terminal and type `python ejecutar_create_dataset.py`.

..* [**ejecutar_pruebas.py.**](../../blob/DQP-allgames/GVGAI/clients/GVGAI-PythonClient/src/ejecutar_pruebas.py) Once the training datasets have been created, this file is able to train the agent on them and, once trained, evaluate it on the test levels. The test results are saved to the [**test_output.txt.**](../blob/DQP-allgames/GVGAI/clients/GVGAI-PythonClient/src/test_output.txt) file. To execute it, open up a terminal and type `python ejecutar_pruebas.py`.
