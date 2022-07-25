import plotly.plotly as py
import plotly 
import plotly.graph_objs as go
from keras.models import model_from_json
plotly.tools.set_credentials_file(username='BrunaFrade', api_key='8ndE5iHBZzjCCpNOBFwn')

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

    X_train_src = np.vstack(df_train_src['image'].values)   # scale pixel values to [0, 1]

    X_train_src = X_train_src.astype('float32')


    df_train_src['image_gt'] = df_train_src['image_gt'].apply(lambda im: np.fromstring(im, sep=' '))
    
    print('source ',df_train_src.count())  # prints the number of values for each column

    df_train_src = df_train_src.dropna()  # drop all rows that have missing values in them

    gt_train_src = np.vstack(df_train_src['image_gt'].values)/255   # scale pixel values to [0, 1]
    # gt_train_src = gt_train_src.astype('float32')

    y_train_src = (df_train_src[df_train_src.columns[:-2]].values )

   
    
   
    # y_train_src = y_train_src.astype('uint8')
    # idx10 = np.where(y_train_src > 32)
    # y_train_src[idx10] = 1
    # print y_train_src
    # y_train_src=y_train_src/32.
    y_train_src=(y_train_src-16)/16.

    X_train_src = X_train_src.reshape(X_train_src.shape[0], 32, 32, 1)
    # gt_train_src = gt_train_src.reshape(gt_train_src.shape[0], 32, 32, 3)
    # y_train_src = y_train_src.reshape(y_train_src.shape[0],1,1,6)
    y_train_src = y_train_src.reshape(y_train_src.shape[0],6)



    X_train_src = preprocess_images(X_train_src, tmin=0, tmax=1)
    print y_train_src[0]
    return(X_train_src,gt_train_src,y_train_src)
def preprocess_images(X, tmin=-1, tmax=1):
    V = X * (tmax - tmin) / 255.
    V += tmin
    return V

def postprocess_images(V, omin=-1, omax=1):
    X = V - omin
    X = X * 255. / (omax - omin)
    return X
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


def graphic_keypoints(ours,ours_error, interspecies,interspecies_error):

    # trace1 = go.Bar(
    #     x=['LE', 'LR', 'N'],
    #     y=ours,
    #     name='Our',
    #     error_y=dict(
    #         type='data',
    #         array=[1.8, 1.5, 2],
    #         visible=True
    #     )
    # )
    # trace2 = go.Bar(
    #     x=['LE', 'LR', 'N'],
    #     y=interspecies,
    #     name='Interspecies',
    #     error_y=dict(
    #         type='data',
    #         array=[1.4, 1.7, 2.5],
    #         visible=True
    #     )
    # )
    # data = [trace1, trace2]
    # layout = go.Layout(
    #     barmode='group'
    # )
    # fig = go.Figure(data=data, layout=layout)
    # # py.iplot(fig, filename='error-bar-bar')
    # py.image.save_as(fig, filename='/home/brunafrade/experimentos/evaluate/comparative_failure_keypoint.png')
    n_groups = 3

     
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8
     
    rects1 = plt.bar(index, ours, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Ours')
     
    rects2 = plt.bar(index + bar_width, interspecies, bar_width,
                     alpha=opacity,
                     color='orange',
                     label='Interspecies')
     
    # plt.xlabel('Keypoints',size=17)
    plt.ylabel('Precision',size=17)
    # plt.title('Scores by person')
    plt.xticks(index + bar_width, ('Left eye', 'Right eye', 'Nose'),size=17)
    plt.legend(bbox_to_anchor=(0.43, 0.9,0.55, .96), loc=3,
       ncol=2, mode="expand", borderaxespad=0.)
     
    plt.tight_layout()
    plt.savefig(filename='/home/brunafrade/experimentos/evaluate/comparative_failure_keypoint.pdf')
    plt.close()
def graphic_compare_number_image(ours,interspecies):

    # # Add data
    # number_image=['10','100','500','1000']
    # #number_image = ['0','10','25']
    # #tif = [32.5, 37.6, 49.9, 53.0, 69.1, 75.4]
    # #interspecies = [13.8, 22.3, 32.5, 37.2, 49.9, 56.1]
    # # ours = [36.5, 26.6, 43.6, 52.3, 71.5, 81.4]

    # area_ours = trapz(ours, dx=5)
    # area_cat = trapz(interspecies, dx=5)



    # # Create and style traces

    # trace1 = go.Scatter(
    #     x = number_image,
    #     y = ours,
    #     name = 'Ours area: %.2f'%area_ours,
    #     line = dict(
    #         color = ('rgb(10, 12, 80)'),
    #         width = 4,
    #         dash = 'dash') # dash options include 'dash', 'dot', and 'dashdot'
    # )
    # trace2 = go.Scatter(
    #     x = number_image,
    #     y = interspecies,
    #     name = 'Interspecies area: %.2f'%area_cat,
    #     line = dict(
    #         color = ('rgb(205, 12, 24)'),
    #         width = 4
    #         ) # dash options include 'dash', 'dot', and 'dashdot'
    # )





    # data = [trace1, trace2]

    # # Edit the layout
    # layout = dict(
    #               xaxis = dict(title = 'Dataset Size (images)'),
    #               yaxis = dict(title = 'Failure Rate %'),
    #               )

    # fig = dict(data=data, layout=layout)
    # # py.iplot(fig, filename='styled-line')
    # py.image.save_as(fig, filename='/home/brunafrade/experimentos/evaluate/comparative_failure_imageNumber.png')
    # data to plot
    area_ours = trapz(ours, dx=5)
    area_interspecies = trapz(interspecies, dx=5)
   
    x=number_image=[10,100,500,1000]
    print len(ours)
    plt.plot( x, ours, 'bo') # green bolinha
    plt.plot( x, ours, 'k:', color='blue',label='Ours  AUC= %.2f'%area_ours) # linha pontilha orange

    plt.plot( x, interspecies, 'r^') # red triangulo
    plt.plot( x, interspecies, 'k--', color='red',label='Interspecies  AUC= %.2f'%area_interspecies,)  # linha tracejada azul

    
    plt.axis([1, 1000,0, 100])
    # plt.title("Mais incrementado")
    # plt.rcParams['legend.fontsize'] = 12
    # plt.rc('axes', titlesize=18)
    # plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=18) 
    plt.rcParams.update({'font.size': 13})
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.xlabel("Dataset size",size=17)
    plt.ylabel("Precision",size=17)
    plt.savefig(filename='/home/brunafrade/experimentos/evaluate/comparative_interspecies_imageNumber.pdf')
    plt.close()



Xr_train,_, Y_train = load('/home/brunafrade/dataset/training_human_32x32.csv')

animal='horse'
Xr_tgt_test,_, Y_tgt_test = load('/home/brunafrade/dataset/training_'+animal+'_32x32.csv')

param=[10,100,500,1000]
failure=[]
failure_ours=[]
failure_interspecies=[]


for i in param:

    #ours
    json_file = open('/home/brunafrade/experimentos/evaluate/weights/compareInterspecies/'+animal+'/ours_horse_tresh_{}_conf.json'.format(i), 'r')
    # json_file = open('/home/brunafrade/experimentos/baseline/cat_conf.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_ours = model_from_json(loaded_model_json)
    model_ours.load_weights('/home/brunafrade/experimentos/evaluate/weights/compareInterspecies/'+animal+'/ours_horse_tresh_{}_weights.h5'.format(i),by_name=True)

    key,l=new_accuracy(model_ours.predict(Xr_tgt_test[1000:]),Y_tgt_test[1000:])
    l=l*100
    failure_ours.append(l)
interspecies=[100-91.55,100-81.29,100-37.91,100-25.97]
graphic_compare_number_image(failure_ours,interspecies)
interspecies=[100-23.39,100-27.13,100-27.41]
graphic_keypoints(key*100,0, interspecies,0)