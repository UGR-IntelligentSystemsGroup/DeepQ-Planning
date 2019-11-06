# Script que lee los resultados de test_output (para los nuevos niveles de test)
# y crea un archivo csv de salida 

import re
import numpy as np

path_input = 'test_output.txt'
path_csv = 'test_output_parseado.csv'

nivel_inicial = 5
nivel_final = 14

pruebas_por_modelo = 6
num_modelos = 15
num_niveles_test = 10

with open(path_input, 'r') as file_input:
	data_input = file_input.read()

	matchs_level_0 = re.findall(r'level 0 - [0-9]+', data_input)
	matchs_level_1 = re.findall(r'level 1 - [0-9]+', data_input)

	# Me quedo solo con el número de acciones
	arr_level_0 = np.array([int(i.split()[-1]) for i in matchs_level_0], dtype='int')
	arr_level_1 = np.array([int(i.split()[-1]) for i in matchs_level_1], dtype='int')

	# Combierto el array en un tensor 3D:
	# arr[nivel_test][id_modelo][id_prueba(según tamaño de dataset)]
	arr_level_0 = arr_level_0.reshape(-1, num_modelos, pruebas_por_modelo)
	arr_level_1 = arr_level_1.reshape(-1, num_modelos, pruebas_por_modelo)

	# Mezclo los dos arrays en uno
	# Primer array level 0, primero level 1, segundo level 0, primero level 1...
	arr_levels = np.empty(shape=(num_niveles_test,num_modelos,pruebas_por_modelo), dtype='int')
	arr_levels[0::2,:,:] = arr_level_0
	arr_levels[1::2,:,:] = arr_level_1

	print(arr_levels[0])

	# Lo guardo en el csv
	with open(path_csv, 'w') as file:
		for i in range(arr_levels.shape[0]):
			file.write('- Nivel {} -\n'.format(nivel_inicial+i))
			np.savetxt(file, arr_levels[i].T, delimiter=',', fmt='%d') # Lo traspongo para que cada modelo sea una columna
			file.write('\n')









