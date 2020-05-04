import pandas as pd
import numpy as np
import csv
"""
read csv file into numpy
"""
para_matrix = []

glaucoma_list = []
with open('D:\\Code\\ROTA_TextureFeature\\dataset\\TF_Wave_Stat_MH_Glaucoma.csv') as f:
	f_csv = csv.reader(f)
	data_glaucoma = []
	for row in f_csv:
		data_glaucoma.append(row)
		print(len(data_glaucoma))

for i in range(1,len(data_glaucoma)):
	glaucoma_list.append(data_glaucoma[i][0])
	for j in range(1,len(data_glaucoma[i])):
		data_glaucoma[i][j] = float(data_glaucoma[i][j])
for i in range(len(data_glaucoma)):
	para_matrix.append(data_glaucoma[i][0:10])
#np.save("D:\\Code\\ROTA_TextureFeature\\dataset\\TF_Wave_Stat_MH_Glaucoma.npy",data_glaucoma)
#np.save("D:\\Code\\ROTA_TextureFeature\\dataset\\glaucoma_list.npy",glaucoma_list)

data_glaucoma = []

normal_list = []
with open('D:\\Code\\ROTA_TextureFeature\\dataset\\TF_Wave_Stat_MH_Healthy.csv') as f:
	f_csv = csv.reader(f)
	data_normal = []
	for row in f_csv:
		data_normal.append(row)
		print(len(data_normal))
for i in range(1,len(data_normal)):
	normal_list.append(data_normal[i][0])
	for j in range(1,len(data_normal[i])):
		data_normal[i][j] = float(data_normal[i][j])
	para_matrix.append(data_normal[i][0:10])
#np.save("D:\\Code\\ROTA_TextureFeature\\dataset\\TF_Wave_Stat_MH_Healthy.npy",data_normal)
#np.save("D:\\Code\\ROTA_TextureFeature\\dataset\\normal_list.npy",normal_list)
data_normal = []
#np.save("D:\\Code\\ROTA_TextureFeature\\dataset\\para_matrix.npy",para_matrix)
print(np.shape(para_matrix))
np.save("D:\\Code\\ROTA_TextureFeature\\dataset\\para_matrix.npy",para_matrix)
