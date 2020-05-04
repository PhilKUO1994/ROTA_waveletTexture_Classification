#from ROTA_waveletTexture_Classification.read_csv import read_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def floating_arr(data):
	for i in range(len(data)):
		for j in range(len(data[i])):
			try:
				data[i][j] = float(data[i][j])
			except ValueError:
				print("")
	return data

def Separating_x_y(data):
	features = []
	for i in range(1, lens):
		features.append("GaborWavelet_1_" + str(i))
	# Separating out the features
	x = data.loc[:, features].values
	# Separating out the target
	y = data.loc[:, ['Target']].values
	x = floating_arr(x)
	y = floating_arr(y)
	return x,y

data = np.load("D:\\Code\\ROTA_TextureFeature\\dataset\\para_matrix_example.npy")
lens = len(data[0])
data[0][0] = "Target"
for i in range(1,51):
	data[i][0] = 1
for i in range(51,len(data)):
	data[i][0] = 0

features = []
features.append("Target")
for i in range(1,lens):
	features.append("GaborWavelet_1_"+str(i))

df = pd.DataFrame(data=data[1:], columns=features)
randomlize_dataset = True
if randomlize_dataset:
	trainset, testset = train_test_split(df,test_size=0.3)
	X_train,y_train = Separating_x_y(trainset)
	X_test, y_test = Separating_x_y(testset)
	print(X_train[0])

	# Standardizing the features
	std_scaler = StandardScaler()
	X_train = std_scaler.fit_transform(X_train)
	X_test = std_scaler.transform(X_test)

	# Perform PCA on standarized data
	pca = PCA(n_components=100)
	X_train = pca.fit_transform(X_train)
	X_test = pca.transform(X_test)
	print(y_train)
	y_train = y_train.astype('int')
	y_test = y_test.astype('int')

	np.save("X_train.npy",X_train)
	np.save("y_train.npy",y_train)
	np.save("X_test.npy",X_test)
	np.save("y_test.npy",y_test)


'''
# Standardizing the features
x = StandardScaler().fit_transform(df)

# Intitialing column names
column_names = []
for i in range(100):
	column_names.append("principal component "+str(i+1))





# Perform PCA on standarized data
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = column_names)
principalDf.to_csv(r"D:\\Code\\ROTA_TextureFeature\\dataset\\principalDf.csv", index = False, header=True)
print(principalDf)
'''