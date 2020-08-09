# -*- coding: utf-8 -*-
"""
This program receives as inputs the names (without the lv index) of the levels
created for Boulder Dash, Ice and Fire or Catapult and the directory that contains
them, and checks that every level is correct.

@author: Carlos Núñez Molina
"""

import glob
import sys

# Choose the game

chosen_game_str = input("Choose the game: 0 -> BoulderDash, 1 -> IceAndFire, 2 -> Catapult. \n")

# Validate the input
try:
    chosen_game = int(chosen_game_str)
    
    if chosen_game < 0 or chosen_game > 2:
        sys.exit("Entrada incorrecta.")
    
except ValueError:
    sys.exit("Entrada incorrecta.")

# Get the directory and the lv names

dir_path = input("Introduce the directory path containing the levels: ")

# Make sure the path ends with '/'
if dir_path[-1] != '/':
    dir_path += '/'

lv_names = input("Introduce the name of the levels, without the index and the file format: ")

# Patterns that the contents of each level must match
# Format: <character>, <number of times it must appear>

# BoulderDash
patterns_list_bd = []
patterns_list_bd.append(('A',1)) # There must be only one agent (A)
patterns_list_bd.append(('e',1)) # There must be only one exit (e)
patterns_list_bd.append(('x',23)) # There must be only 23 gems (x)

# IceAndFire
patterns_list_iaf = []
patterns_list_iaf.append(('A',1)) # There must be only one agent (A)
patterns_list_iaf.append(('e',1)) # There must be only one exit (e)
patterns_list_iaf.append(('f',1)) # There must be only one fire boot (f)
patterns_list_iaf.append(('i',1)) # There must be only one ice boot (i)
patterns_list_iaf.append(('c',10)) # There must be only 10 coins (c)

# Catapults
patterns_list_cat = []
patterns_list_iaf.append(('A',1)) # There must be only one agent (A)
patterns_list_iaf.append(('g',1)) # There must be only one exit (g)

# Use the correct patterns according to the chosen game
if chosen_game == 0: # BoulderDash
    patterns_list = patterns_list_bd
elif chosen_game == 1: # IceAndFire
    patterns_list = patterns_list_iaf
else: # Catapults
    patterns_list = patterns_list_cat

# Use glob to get all the levels
lv_paths = glob.glob(dir_path + lv_names + '*')

if lv_paths == []:
    print("No levels found!")

for curr_lv in lv_paths:
    # Open the current level and read it
    with open(curr_lv, 'r') as f:
        curr_lv_contents = f.read()
        
        # Make sure it matches every pattern
        lv_format_is_correct = True
        
        for char, num_rep in patterns_list:
            # Count the number of occurrences of char and see if it is num_rep
            if curr_lv_contents.count(char) != num_rep:
                lv_format_is_correct = False
                
        if not lv_format_is_correct:
            print("El nivel {} es incorrecto!!".format(curr_lv))
        

