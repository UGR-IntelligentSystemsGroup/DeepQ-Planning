"""
@author: Carlos Núñez Molina

GVGAI Level Editor
"""

import tkinter as tk
from tkinter import W, E, S, N
from tkinter import simpledialog
import os
import numpy as np
import re

# Creates the tiles of the level, given by n_rows and n_cols
# If create_new is True, all the map is initialized to the color given by
# selected_tile_color. If it's false, the map is drawn according to the contents
# of tile_array.
def create_tiles(n_rows=13, n_cols=26, create_new=True):
    global tile_array
    
    canvas_height = map_canvas.winfo_height()  
     
    global tile_height # Make the variables global so that it can be accessed from change_tile()
    tile_height = canvas_height // n_rows
    global tile_width # The width is equal to the height, so the tiles are squared
    tile_width = tile_height
    
    # Create all the tiles according to n_rows and n_cols
    for curr_row in range(n_rows):
        for curr_col in range(n_cols):
            x_ini = curr_col*tile_width
            y_ini = curr_row*tile_height
            x_fin = (curr_col+1)*tile_width
            y_fin = (curr_row+1)*tile_height
            
            # Draw the left and top outline
            if x_ini == 0:
                x_ini = 2
            if y_ini == 0:
                y_ini = 2
            
            if create_new:
                map_canvas.create_rectangle(x_ini, y_ini, x_fin, y_fin,
                                            fill=color_array[selected_tile_color],
                                            outline="black")
            else:
                map_canvas.create_rectangle(x_ini, y_ini, x_fin, y_fin,
                            fill=color_array[tile_array[curr_row][curr_col]],
                            outline="black")
            
    # Tile Array
    # It contains the type of each tile
    if create_new:
        tile_array = [[selected_tile_color for a in range(n_cols)] for b in range(n_rows)]
    
    # Shrink the map canvas so that it perfectly fits the size filled by the tiles
    new_height  = tile_height*n_rows
    new_width = tile_width*n_cols
    map_canvas.config(width=new_width-1, height=new_height-1)

# Called when the mouse is clicked over a tile. It changes its type to the
# one currently selected.
def change_tile(event):
    # Get the corresponding tile from the position (x, y) of the mouse click
    tile_row = event.y // tile_height
    tile_col = event.x // tile_width
    
    # Make sure tile_row and tile_col are not out of range
    tile_row = min(tile_row, len(tile_array)-1)
    tile_col = min(tile_col, len(tile_array[0])-1)
    
    # Change the color of the tile
    x_ini = tile_col*tile_width
    y_ini = tile_row*tile_height
    x_fin = (tile_col+1)*tile_width
    y_fin = (tile_row+1)*tile_height
    
    # Draw the left and top outline
    if x_ini == 0:
        x_ini = 2
    if y_ini == 0:
        y_ini = 2
    
    map_canvas.create_rectangle(x_ini, y_ini, x_fin, y_fin,
                                    fill=color_array[selected_tile_color],
                                    outline="black")
    
    # Change the type of the tile
    tile_array[tile_row][tile_col] = selected_tile_color

# Creates the upper part of the window, which shows the level map
def create_level_grid(root):
    global map_canvas # It needs to be accessed from change_tile()
    
    map_canvas = tk.Canvas(root)
    map_canvas.config(width=600, height=400, bg="white")
    map_canvas.grid(row=0, column=0, padx=10, pady=10) # Upper Middle of the window
    map_canvas.update() # Update frame size
    
    # Paint all the tiles/cells of the level
    create_tiles()
    
    # Associate mouse events
    map_canvas.bind("<Button-1>", change_tile)
    map_canvas.bind("<B1-Motion>", change_tile)

# Creates the grid used to associate a tile type (character) with a color  
def create_type_selection_grid(frame):
    # Colors to use for the color tiles
    global color_array
    color_array = ['yellow', 'red', 'green', 'blue', 'pink', 
                   'orange', 'dark violet', 'brown', 'cyan', 'gray']
    
    # Create color tiles
    color_tiles_array = []
    
    for curr_row in range(2):
        for curr_col in range(5):
            curr_canvas = tk.Canvas(frame, bg=color_array[curr_row*5 + curr_col],
                                    width=60, height=60)
            curr_canvas.grid(row=curr_row*2 + 1, column=curr_col)
            color_tiles_array.append(curr_canvas)
      
    # Associate a mouse event to each color tile that when the user clicks on it,
    # selected_tile_color changes to the color of the tile    
    
    # This can't be done with a for-loop because the iterator variable is shared
    # among the different lambda functions and all the functions are called
    # with the value color_array[i], where i would always store the last value
    # (9)
    color_tiles_array[0].bind("<Button-1>", lambda event: set_selected_tile_color(0))
    color_tiles_array[1].bind("<Button-1>", lambda event: set_selected_tile_color(1))
    color_tiles_array[2].bind("<Button-1>", lambda event: set_selected_tile_color(2))
    color_tiles_array[3].bind("<Button-1>", lambda event: set_selected_tile_color(3))
    color_tiles_array[4].bind("<Button-1>", lambda event: set_selected_tile_color(4))
    color_tiles_array[5].bind("<Button-1>", lambda event: set_selected_tile_color(5)) 
    color_tiles_array[6].bind("<Button-1>", lambda event: set_selected_tile_color(6))
    color_tiles_array[7].bind("<Button-1>", lambda event: set_selected_tile_color(7))
    color_tiles_array[8].bind("<Button-1>", lambda event: set_selected_tile_color(8))
    color_tiles_array[9].bind("<Button-1>", lambda event: set_selected_tile_color(9)) 
   
    # Stores the character (type) associated with each color tile
    global type_array
    type_array = [tk.StringVar() for a in range(10)]
    
    # Create Entries where the user will introduce the character associated
    # with each color
    for curr_row in range(2):
        for curr_col in range(5):
            curr_entry = tk.Entry(frame, textvariable=type_array[curr_row*5 + curr_col],
                                  width=2)
            curr_entry.grid(row=curr_row*2, column=curr_col)
    
# Creates the lower part of the window, which shows the mapping color-tile type    
def create_colors_grid(root):
    frame = tk.Frame(root)
    frame.config(height=150) 
    frame.grid(row=1, column=0, padx=10, pady=10, sticky=S) # Lower Middle of the window
    frame.update() # Update frame size
    
    # Create the grid that shows the mapping between colors and tile types
    create_type_selection_grid(frame)
    
    # Selected Tile Color
    # This is the color a tile will be changed to when the user clicks on it
    # Number "i" means color_array[i]
    global selected_tile_color
    selected_tile_color = 0
    
# Change the selected_tile_color
def set_selected_tile_color(new_color):
    global selected_tile_color # Refer to the global variable
    selected_tile_color = new_color   
    
# It stores the current map given by the tiles drawn in the map_canvas to
# the filepath given by map_filename and map_directory
def save_current_map():        
    # Get the character associated with each color tile
    # Note: a color tile can have more than one character associated,
    # although in boulder dash each observation type only has one character
    # associated
    character_mapping = []
    default_char = '?' # If a color tile has no character associated, this one will be used
    
    for chars in type_array:
        curr_chars = chars.get()
        
        if curr_chars == '':
            character_mapping.append(default_char)
        else:
            character_mapping.append(curr_chars)
        
    # Save the current map to the file given by map_filename and map_directory
    if map_filename == '':
        filename = "new_level.txt"
    else:
        filename = map_filename
    
    # If the directory doesn't exist, create it
    if not os.path.exists(map_directory):
        os.makedirs(map_directory)
    
    filepath = map_directory + '/' + filename
    
    with open(filepath, 'w') as f:
        for curr_row in range(len(tile_array)):
            for curr_col in range(len(tile_array[0])):
                f.write(character_mapping[tile_array[curr_row][curr_col]])
            
            f.write('\n')
            
    tk.messagebox.showinfo("File Saved", "The level {} has been saved.".format(filepath))

# Loads the map given by map_filename and map_directory and draws it in the map_canvas
# according to the current tile_type-color mapping
def load_map():
    # Load the current map and read it
    if map_filename == '':
        tk.messagebox.showerror("Loading Error", "Empty filename.")
        return
    
    filepath = map_directory + '/' + map_filename
    
    # Check if the map path exists
    if not os.path.exists(filepath):
        tk.messagebox.showerror("Loading Error", "The map doesn't exist.")
        return
    
    map_string = ""
    map_cols = 0 # Number of columns of the loaded map
    map_rows = 0 # Number of rows of the loaded map
    
    with open(filepath, 'r') as f:
        map_cols = len(f.readline()) - 1 # Calculate the map cols as the size of a line of the file
        
        f.seek(0)
        map_string = f.read() # Read the complete file
        # Calculate the map rows as the number of lines of the file
        map_rows = round(len(map_string) / (map_cols+1)) # Use round because we don't know if the last line of the file ends with '\n' or not
        
    # Format the map contents as a matrix
    map_char_matrix = np.array([char for char in map_string if char != '\n']).reshape((map_rows, map_cols))
    
    # Convert the map contents from chars to integers, according to the current mapping tile_type-color
    
    # Current contenst of type_array
    chars_array = [string_var.get()  if string_var.get() != '' else '?' for string_var in type_array]
    
    map_int_matrix = np.empty(shape=(map_rows, map_cols), dtype=np.int)
    
    for curr_row in range(map_rows):
        for curr_col in range(map_cols):
            curr_elem = map_char_matrix[curr_row, curr_col]
            
            # The corresponding integer (color tile) of the curr_elem char
            # is the position of curr_elem in chars_array
            ind = chars_array.index(curr_elem)
            map_int_matrix[curr_row, curr_col] = ind
            
    # Save contents in the tile_array
    global tile_array
    tile_array = map_int_matrix.tolist()       
            
    # Draw the new map
    create_tiles(n_rows=map_rows, n_cols=map_cols, create_new=False)
    
    tk.messagebox.showinfo("File Loaded", "The level {} has been successfully loaded.".format(filepath))
    
# Sets the filename of the map to load/save
def set_map_filename():
    global map_filename
    
    new_filename = simpledialog.askstring("Map File Name", "Introduce the new file name",
                                          initialvalue=map_filename)
    
    if new_filename is not None:
        map_filename = new_filename

# Sets the directory where the maps will be saved/loaded from
def set_map_directory():
    global map_directory
    
    new_directory = simpledialog.askstring("Maps Directory",
                                           "Introduce the new directory (without the last '/') where the maps will be saved/loaded from",
                                           initialvalue=map_directory)
    
    if new_directory is not None:
        map_directory = new_directory
  
# Changes the number of rows and columns of the map and redraws it.
# The contents of the current map will be lost!!
def change_map_size():    
    s = simpledialog.askstring("Map Size", "Introduce the rows and cols of the new map, separated by a space")
    
    # If the user didn't introduce anything, do nothing and exit the function
    if s is None:
        return
    
    # Check if the format is correct
    if re.match("[0-9]* [0-9]*", s) is None:
        tk.messagebox.showerror("Input Error", "Wrong format.")
        return
    
    # Obtain the number of rows and columns as introduced by the user
    num_rows = int(re.search('[0-9]* ', s).group(0)[:-1])
    num_cols = int(re.search(' [0-9]*', s).group(0)[1:])
    
    # Redraw the map according to the new size
    # Note: the contents of the current map will be lost!!
    create_tiles(n_rows=num_rows, n_cols=num_cols)
    
# Creates the menu bar, used to save and load maps
def create_menu(root):
    menubar = tk.Menu(root)
    root.config(menu=menubar)
    
    # Map size option
    menubar.add_command(label="Map Size", command=change_map_size)
    
    # Save option
    menubar.add_command(label="Save", command=save_current_map)

    # Create load option
    menubar.add_command(label="Load", command=load_map)
    
    # Filename option
    global map_filename # Accessed from set_map_filepath and save_current_map
    map_filename = ""
    menubar.add_command(label="Change name", command=set_map_filename)
    
    # Directory option
    global map_directory # Accessed from set_map_directory and save_current_map
    map_directory = "."
    menubar.add_command(label="Change directory", command=set_map_directory)
    

def main():
    width = 1000 # Screen size
    height = 600
    
    root = tk.Tk()
    root.title("GVGAI Level Editor")
    root.geometry("{}x{}".format(width, height))
   
    root.columnconfigure(0, weight=1) # Make the child frames resize with the window
    root.rowconfigure(0, weight=2)
    root.rowconfigure(1, minsize=150, weight=1)
    
    create_colors_grid(root)
    create_level_grid(root)
    create_menu(root)
    
    root.mainloop()
    
      
if __name__ == "__main__":
    main()