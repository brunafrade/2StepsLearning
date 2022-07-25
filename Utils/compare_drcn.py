# coding=utf8
import sys  

reload(sys)  
sys.setdefaultencoding('utf8')
from keras.models import model_from_json

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

def new_accuracy(keypoint_pred, keypoint_gt):
    keypoint_pred=keypoint_pred.reshape(keypoint_pred.shape[0],6)
    keypoint_gt=keypoint_gt.reshape(keypoint_gt.shape[0],6)
    # Tamanhos incompativeis
    if keypoint_pred.shape != keypoint_gt.shape:
        return

    # Coordenadas faltando
    if (keypoint_pred.shape[1] % 2 != 0) | (keypoint_gt.shape[1] % 2 != 0):
        return
    # print keypoint_pred
    THRESHOLD = 0.1
    FACE_SIZE = 2
    FAILURE_COUNT = 0
    FAILURE_COUNT_keypoint=np.zeros(3)
    euc_dist = np.zeros((keypoint_gt.shape[0], keypoint_gt.shape[1]/2))

    # Supondo que os pontos estao ordenados em X, Y, X, Y, X, Y...
    for i in range(keypoint_gt.shape[1]/2):
        euc_dist[:, i] = np.sqrt((keypoint_pred[:, 2*i] - keypoint_gt[:, 2*i])**2 + (keypoint_pred[:, 2*i + 1] - keypoint_gt[:, 2*i + 1])**2)
        
        for j in range(keypoint_gt.shape[0]):
            if euc_dist[j, i] > (THRESHOLD * FACE_SIZE):
                FAILURE_COUNT += 1
                FAILURE_COUNT_keypoint[i] +=1
    AVG_FAILURE_RATE = float(FAILURE_COUNT) / (keypoint_gt.shape[0] * keypoint_gt.shape[1]/2)
    FAILURE_COUNT_keypoint = FAILURE_COUNT_keypoint/keypoint_gt.shape[0]
    return 1-FAILURE_COUNT_keypoint,1-AVG_FAILURE_RATE
def plot_sample(x, y,axis):
    # y=y.reshape(6)
    img = x.reshape(32, 32)
    axis.imshow(img,cmap='gray')
    axis.scatter(y[0]*16 , y[1]*16 , marker='<', s=5)
    axis.scatter(y[2]*16 , y[3]*16 , marker='>', s=5)
    axis.scatter(y[4]*16 , y[5]*16 , marker='o', s=5)


def visualization (model, x_heat,x,PARAMDIR,image_name,animal):
# visualization 
        
    Xs = x_heat[:100]
    # Xs = x
    # print Xs
    # Xs = postprocess_images(Xs, omin=0, omax=1)
    # Xs = np.reshape(Xs, (len(Xs), Xs.shape[3], Xs.shape[1], Xs.shape[2]))
    # #show_images(Xs, filename=image_name)             
    # Xs_pred = drcn.convae_model.predict(Xs)
    # Xs_pred = postprocess_images(Xs_pred, omin=0, omax=1)
    # Xs_pred = np.reshape(Xs_pred, (len(Xs_pred), Xs_pred.shape[3], Xs_pred.shape[1], Xs_pred.shape[2]))
    # imgfile = PARAMDIR+'saida_autoencoder/o_'+image_name
    # show_images(Xs, filename=imgfile)
    # imgfile = PARAMDIR+'saida_autoencoder/p_'+image_name

    # show_images(Xs_pred, filename=imgfile)

    y_pred = model.predict(x_heat[:100])
    Xz = postprocess_images(x_heat[:100], omin=0, omax=1)
    Xz = np.reshape(Xz, (len(Xz), Xz.shape[3], Xz.shape[1], Xz.shape[2]))
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(Xs.shape[0]):
        ax = fig.add_subplot(np.sqrt(Xs.shape[0]), np.sqrt(Xs.shape[0]), i + 1, xticks=[], yticks=[])
        plot_sample(Xz[i], y_pred[i], ax)

    fig.savefig("/home/brunafrade/experimentos/evaluate/images/"+animal+"/"+image_name)

    plt.close()
def preprocess_images(X, tmin=-1, tmax=1):
    V = X * (tmax - tmin) / 255.
    V += tmin
    return V

def postprocess_images(V, omin=-1, omax=1):
    X = V - omin
    X = X * 255. / (omax - omin)
    return X

def load(FTRAIN_source):
    df_train_src = read_csv(os.path.expanduser(FTRAIN_source))  # load pandas dataframe
    df_train_src['image'] = df_train_src['image'].apply(lambda im: np.fromstring(im, sep=' '))
    print('source ',df_train_src.count())  # prints the number of values for each column
    df_train_src = df_train_src.dropna()  # drop all rows that have missing values in them

    X_train_src = np.vstack(df_train_src['image'].values)/255.   # scale pixel values to [0, 1]
    X_train_src = X_train_src.astype('float32')

    df_train_src['image_gt'] = df_train_src['image_gt'].apply(lambda im: np.fromstring(im, sep=' '))
    print('source ',df_train_src.count())  # prints the number of values for each column
    df_train_src = df_train_src.dropna()  # drop all rows that have missing values in them

    gt_train_src = np.vstack(df_train_src['image_gt'].values)/255   # scale pixel values to [0, 1]
   
    y_train_src = (df_train_src[df_train_src.columns[:-2]].values )
    y_train_src=(y_train_src-16)/16.
    
    X_train_src = X_train_src.reshape(X_train_src.shape[0], 32, 32, 1)
    y_train_src = y_train_src.reshape(y_train_src.shape[0],6)

    return(X_train_src,gt_train_src,y_train_src)

def graphic_keypoints(ours,ours_error, drcn,drcn_error):


    n_groups = 3

     
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8
     
    rects1 = plt.bar(index, ours, bar_width,
                     alpha=opacity,
                     color='b',
                     label='2Steps')
     
    rects2 = plt.bar(index + bar_width, drcn, bar_width,
                     alpha=opacity,
                     color='orange',
                     label='DRCN')
     
    # plt.xlabel('Keypoints',size=17)
    plt.ylabel('Precisão',size=17)
    # plt.title('Scores by person')
    plt.xticks(index + bar_width, ('Olho Esquerdo', 'Olho Direito', 'Nariz'),size=17)
    plt.legend(bbox_to_anchor=(0.43, 0.9,0.55, .96), loc=3,
       ncol=2, mode="expand", borderaxespad=0.)
     
    plt.tight_layout()
    plt.savefig(filename='/home/brunafrade/experimentos/evaluate/comparative_failure_keypoint_drcn.pdf')
    plt.close()

def graphic_compare_number_image(ours,drcn,name):

 
    area_ours = trapz(ours, dx=5)
    area_drcn = trapz(drcn, dx=5)
   
    x=number_image=[0,10,50,100,500,1000]
    print len(ours)
    plt.plot( x, ours, 'bo') # green bolinha
    plt.plot( x, ours, 'k:', color='blue',label='2Steps  AUC= %.2f'%area_ours) # linha pontilha orange

    plt.plot( x, drcn, 'r^') # red triangulo
    plt.plot( x, drcn, 'k--', color='red',label='DRCN  AUC= %.2f'%area_drcn,)  # linha tracejada azul

    
    plt.axis([1, 1000,0, 100])
    # plt.title("Mais incrementado")
    # plt.rcParams['legend.fontsize'] = 12
    # plt.rc('axes', titlesize=18)
    # plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=18) 
    plt.rcParams.update({'font.size': 13})
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.xlabel("Quantidade de Imagens",size=17)
    plt.ylabel("Precisão",size=17)
    plt.savefig(filename='/home/brunafrade/experimentos/evaluate/sourceHuman_comparative_drcn_imageNumber_'+name+'.pdf')
    plt.close()


Xr_train,_, Y_train = load('/home/brunafrade/dataset/training_human_32x32.csv')

animals=['horse','cat','dog']
for animal in animals:
    Xr_tgt_test,_, Y_tgt_test = load('/home/brunafrade/dataset/training_'+animal+'_32x32.csv')

    param=[0,10,50,100,500,1000]
    failure=[]
    failure_ours=[]
    failure_interspecies=[]
    failure_drcn = []


    for i in param:

        #ours
        json_file = open('/home/brunafrade/experimentos/evaluate/weights/compareInterspecies/'+animal+'/ours_horse_tresh_{}_conf.json'.format(i), 'r')
        # json_file = open('/home/brunafrade/experimentos/evaluate/weights/'+animal+'/ours_tresh_{}_conf.json'.format(i), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model_ours = model_from_json(loaded_model_json)
        model_ours.load_weights('/home/brunafrade/experimentos/evaluate/weights/compareInterspecies/'+animal+'/ours_horse_tresh_{}_weights.h5'.format(i),by_name=True)

        key,l=new_accuracy(model_ours.predict(Xr_tgt_test[1000:]),Y_tgt_test[1000:])
        l=l*100
        failure_ours.append(l)


         #drcn
        json_file = open('/home/brunafrade/experimentos/evaluate/weights/'+animal+'/drcn_tresh_{}_cnn_conf.json'.format(i), 'r')
        # json_file = open('/home/brunafrade/experimentos/baseline/cat_conf.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model_ours = model_from_json(loaded_model_json)
        model_ours.load_weights('/home/brunafrade/experimentos/evaluate/weights/'+animal+'/drcn_tresh_{}_cnn_weights.h5'.format(i),by_name=True)

        key_drcn,l=new_accuracy(model_ours.predict(Xr_tgt_test[1000:]),Y_tgt_test[1000:])
        l=l*100
        failure_drcn.append(l)
    #plot graphic
    graphic_compare_number_image(failure_ours,failure_drcn,animal)
    graphic_keypoints(key*100,0, key_drcn*100,0)