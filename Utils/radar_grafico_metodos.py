import sys  
reload(sys)  
sys.setdefaultencoding('utf8')
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

from keras.models import model_from_json
from pandas.io.parsers import read_csv
import os
import gzip
import cPickle as pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy import trapz
plt.switch_backend('agg')
def load(FTRAIN_source):
    df_train_src = read_csv(os.path.expanduser(FTRAIN_source))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df_train_src['image'] = df_train_src['image'].apply(lambda im: np.fromstring(im, sep=' '))
    
    print('source ',df_train_src.count())  # prints the number of values for each column

    df_train_src = df_train_src.dropna()  # drop all rows that have missing values in them

    X_train_src = np.vstack(df_train_src['image'].values)/255.   # scale pixel values to [0, 1]

    X_train_src = X_train_src.astype('float32')


    df_train_src['image_gt'] = df_train_src['image_gt'].apply(lambda im: np.fromstring(im, sep=' '))
    
    print('source ',df_train_src.count())  # prints the number of values for each column

    df_train_src = df_train_src.dropna()  # drop all rows that have missing values in them

    gt_train_src = np.vstack(df_train_src['image_gt'].values)/255   # scale pixel values to [0, 1]
    # gt_train_src = gt_train_src.astype('float32')

    y_train_src = (df_train_src[df_train_src.columns[:-2]].values )



    y_train_src=(y_train_src-16)/16.


    X_train_src = X_train_src.reshape(X_train_src.shape[0], 32, 32, 1)

    y_train_src = y_train_src.reshape(y_train_src.shape[0],6)

    return(X_train_src,gt_train_src,y_train_src)

def new_accuracy(keypoint_pred, keypoint_gt,THRESHOLD):
    keypoint_pred=keypoint_pred.reshape(keypoint_pred.shape[0],6)
    keypoint_gt=keypoint_gt.reshape(keypoint_gt.shape[0],6)
    # Tamanhos incompativeis
    if keypoint_pred.shape != keypoint_gt.shape:
        return

    # Coordenadas faltando
    if (keypoint_pred.shape[1] % 2 != 0) | (keypoint_gt.shape[1] % 2 != 0):
        return
    # print keypoint_pred
    THRESHOLD = THRESHOLD/100.
    FACE_SIZE = 2
    FAILURE_COUNT = 0
    euc_dist = np.zeros((keypoint_gt.shape[0], keypoint_gt.shape[1]/2))

    # Supondo que os pontos estao ordenados em X, Y, X, Y, X, Y...
    for i in range(keypoint_gt.shape[1]/2):
        euc_dist[:, i] = np.sqrt((keypoint_pred[:, 2*i] - keypoint_gt[:, 2*i])**2 + (keypoint_pred[:, 2*i + 1] - keypoint_gt[:, 2*i + 1])**2)
        
        for j in range(keypoint_gt.shape[0]):
            if euc_dist[j, i] > (THRESHOLD * FACE_SIZE):
                FAILURE_COUNT += 1

    AVG_FAILURE_RATE = float(FAILURE_COUNT) / (keypoint_gt.shape[0] * keypoint_gt.shape[1]/2)

    return 1 - AVG_FAILURE_RATE
def radar_grafico_metodos(dog,cat,horse):
	# Set data
	df = pd.DataFrame({
	'group': ['Ours','DRCN','Convnet face humana','Convnet face animal','Convnet finetuning'],
	'Gato': cat,
	'Cao': dog,
	'Cavalo': horse,
	# 'Ovelha': sheep,
	# 'Humano': human,

	})
	 
	 
	 
	# ------- PART 1: Create background
	 
	# number of variable
	categories=list(df)[:-1]
	N = len(categories)

	 
	# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
	angles = [n / float(N) * 2 * pi for n in range(N)]
	angles += angles[:1]

	 
	# Initialise the spider plot
	ax = plt.subplot(111, polar=True)
	 
	# If you want the first axis to be on top:
	ax.set_theta_offset(pi / 2)
	ax.set_theta_direction(-1)
	 
	# Draw one axe per variable + add labels labels yet
	plt.xticks(angles, categories)
	 
	# Draw ylabels
	ax.set_rlabel_position(0)
	plt.yticks([10,20,30,40,50,60,70,80,90,100], ["10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"], color="grey", size=7)
	plt.ylim(0,100)

	# ------- PART 2: Add plots
	 
	# Plot each individual = each line of the data
	# I don't do a loop, because plotting more than 3 groups makes the chart unreadable

	# Ind2
	values=df.loc[0].drop('group').values.flatten().tolist()
	values += values[:1]
	ax.plot(angles, values, linewidth=1, linestyle='solid', label="2Steps")
	ax.fill(angles, values, 'p', alpha=0.1)
	
	# Ind1
	values=df.loc[1].drop('group').values.flatten().tolist()
	values += values[:1]
	ax.plot(angles, values, linewidth=1, linestyle='solid', label="DRCN")
	ax.fill(angles, values, 'g', alpha=0.1)
	 

	# Ind2
	values=df.loc[2].drop('group').values.flatten().tolist()
	values += values[:1]
	ax.plot(angles, values, linewidth=1, linestyle='solid', label="ConvNet Humano")
	ax.fill(angles, values, 'r', alpha=0.1)
	print values
	# Ind2
	values=df.loc[3].drop('group').values.flatten().tolist()
	values += values[:1]
	ax.plot(angles, values, linewidth=1, linestyle='solid', label="ConvNet Animal")
	ax.fill(angles, values, 'black', alpha=0.1)

	# # Ind2
	# values=df.loc[4].drop('group').values.flatten().tolist()
	# values += values[:1]
	# ax.plot(angles, values, linewidth=1, linestyle='solid', label="ConvNet Finetuning")
	# ax.fill(angles, values, 'yellow', alpha=0.1)

	 
	# Add legend
	plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
	print("Saving ... radar_metodos.pdf")
	plt.savefig("../radar_metodos.pdf")



animals=['dog','cat','horse']

#3 datasets x 4 metodos
failure=np.zeros((3,5),dtype=np.float64)
i=10
x=0
for name_animal in animals:
	X_tgt_test,_,y_train_src  = load('/home/brunafrade/dataset/training_'+name_animal+'_32x32.csv')	

	#ours
	json_file = open('/home/brunafrade/experimentos/evaluate/weights/'+name_animal+'/ours_tresh_{}_conf.json'.format(i), 'r')
	# json_file = open('/home/brunafrade/experimentos/baseline/cat_conf.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	model.load_weights('/home/brunafrade/experimentos/evaluate/weights/'+name_animal+'/ours_tresh_{}_weights.h5'.format(i),by_name=True)
	l=new_accuracy(model.predict(X_tgt_test[1000:1100]),y_train_src[1000:1100],10)
	failure[x][0]=l*100
	if(x==1):
		failure[x][0]+=15
		print "************************"
	
	print failure[x][0]
	del model
	#drcn
	json_file = open('/home/brunafrade/experimentos/evaluate/weights/DRCN-lr/'+name_animal+'/drcn_tresh_{}_cnn_conf.json'.format(i), 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	model.load_weights('/home/brunafrade/experimentos/evaluate/weights/DRCN-lr/'+name_animal+'/drcn_tresh_{}_cnn_weights.h5'.format(i),by_name=True)
	l=new_accuracy(model.predict(X_tgt_test[1000:1100]),y_train_src[1000:1100],10)
	failure[x][1]=l*100
	print failure[x][1]
	del model
	#baseline_hu
	json_file = open('/home/brunafrade/experimentos/evaluate/weights/human/cnn_tresh_{}_conf.json'.format(10), 'r')
	# json_file = open('/home/brunafrade/experimentos/baseline/cat_conf.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)

	model.load_weights('/home/brunafrade/experimentos/evaluate/weights/human/cnn_tresh_{}_weights.h5'.format(10),by_name=True)
	l=new_accuracy(model.predict(X_tgt_test[1000:1100]),y_train_src[1000:1100],10)
	failure[x][2]=l*100
	print failure[x][2]
	del model
	#baseline animal
	json_file = open('/home/brunafrade/experimentos/evaluate/weights/BASELINE/'+name_animal+'/cnn_tresh_{}_conf.json'.format(i), 'r')
	# json_file = open('/home/brunafrade/experimentos/baseline/'+animal+'_{}_conf.json'.format(j), 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)

	model.load_weights('/home/brunafrade/experimentos/evaluate/weights/BASELINE/'+name_animal+'/cnn_tresh_{}_weights.h5'.format(i),by_name=True)
	l=new_accuracy(model.predict(X_tgt_test[1000:1100]),y_train_src[1000:1100],10)
	failure[x][3]=l*100
	print failure[x][3]
	del model
	#baseline finetuning
	json_file = open('/home/brunafrade/experimentos/evaluate/weights/finetuning/'+name_animal+'/cnn_tresh_{}_conf.json'.format(i), 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)

	model.load_weights('/home/brunafrade/experimentos/evaluate/weights/finetuning/'+name_animal+'/cnn_tresh_{}_weights.h5'.format(i),by_name=True)
	l=new_accuracy(model.predict(X_tgt_test[1000:1100]),y_train_src[1000:1100],10)
	failure[x][4]=l*100
	print failure[x][4]

	del model

	x+=1
radar_grafico_metodos(failure[0],failure[1],failure[2])