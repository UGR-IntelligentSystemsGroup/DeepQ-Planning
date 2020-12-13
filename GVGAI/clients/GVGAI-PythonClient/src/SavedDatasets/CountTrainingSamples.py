"""
Reads Dataset Sizes.txt and calculates the size (num samples) of the dataset for each gaame.
"""

import numpy as np
import re

games = ['BoulderDash', 'IceAndFire', 'Catapults']

with open("Datasets Sizes.txt", "r") as f:
	file_cont = f.read()

	for game in games:
		matches = re.findall('{}.*'.format(game), file_cont)

		# Get a list with the num samples of each level
		samples_each_level = [int(re.search('[0-9]+$', match).group(0)) for match in matches]

		print(">Samples {} = {}".format(game, np.sum(samples_each_level)))
