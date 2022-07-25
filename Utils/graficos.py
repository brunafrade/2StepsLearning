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



    y_train_src=(y_train_src-16)/16.


    X_train_src = X_train_src.reshape(X_train_src.shape[0], 32, 32, 1)

    y_train_src = y_train_src.reshape(y_train_src.shape[0],6)

    return(X_train_src,gt_train_src,y_train_src)
def graphic_compare_treshold(ours,baseline_cat,baseline_hum,name,folder):
    s=23
    # Add data
    tresh = ['0%', '5%', '10%', '15%', '20%', '25%','30%','35%','40','45','50','55','60','65','70','75','80','85','90','95','100']
    #tif = [100, 30.17, 5.8, 2.3, 1.3, 0.9]
    #interspecies = [100, 30.17, 5.8, 2.3, 1.3, 0.9]
    # ours = [36.5, 26.6, 43.6, 52.3, 71.5, 81.4]
    area_ours = trapz(ours, dx=5)
    area_cat = trapz(baseline_cat, dx=5)
    area_hum = trapz(baseline_hum, dx=5)

    # Create and style traces

    trace1 = go.Scatter(
        x = tresh,
        y = ours,
        name = 'Ours',
        # name = 'Ours A= %.2f'%area_ours,
        line = dict(
            color = ('rgb(10, 12, 80)'),
            width = 4,
            dash = 'dash')
    )

    trace2 = go.Scatter(
        x = tresh,
        y = baseline_hum,
        # name = 'Train Hum Test '+name+'\n A= %.2f'%area_hum,
        name = 'Train Hum Test '+name+'   ',
        line = dict(
            color = ('rgb(22, 96, 167)'),
            width = 4,
            dash = 'dot')
    )
    trace3 = go.Scatter(
        x = tresh,
        y = baseline_cat,
        # name = 'Train '+name+' Test '+name+' A= %.2f'%area_cat,
        name = 'Train '+name+' Test '+name+'   ',
        line = dict(
            color = ('rgb(250, 0, 0)'),
            width = 4,
            dash = 'dot')
    )
    data = [trace1, trace2,trace3]

    # Edit the layout
    layout = dict(legend=dict(
        x=0.5,
        width = 4,



        font=dict(
            family='sans-serif',
            size = s,
            color='#000'
        ),),
        
        xaxis = dict(title = 'Treshold %',
        font=dict(
            family='sans-serif',
            size = s,
            color='#000'
        ),),
        yaxis = dict(title = 'Failure Rate %',
        font=dict(
            family='sans-serif',
            size = s,
            color='#000'
        ),),
                  
                  )

    fig = dict(data=data, layout=layout,font=dict(
            family='sans-serif',
            size = s,
            color='#000'
        ),)
    # py.iplot(fig, filename='styled-line')
    fig.savefig(filename='/home/brunafrade/experimentos/evaluate/graficos/'+folder+'/comparative_failure_tresh.svg')
    # py.image.save_as(fig, filename='/home/brunafrade/experimentos/evaluate/graficos/'+folder+'/comparative_failure_tresh.svg')

def graphic_compare_number_image_matplot(ours,baseline_cat, baseline_hum,regressor_cat,regressor_hum,name,folder):
    x = [0,10,25, 50,100]
    area_ours = trapz(ours, dx=5)
    area_cat = trapz(baseline_cat, dx=5)
    area_hum = trapz(baseline_hum, dx=5)
   
    print len(ours)
    plt.plot( x, ours, 'bo') # green bolinha
    plt.plot( x, ours, 'k:', color='blue',label='Ours  AUC= %.2f'%area_ours) # linha pontilha orange

    plt.plot( x, baseline_cat, 'r^') # red triangulo
    plt.plot( x, baseline_cat, 'k--', color='red',label='Train '+name+' Test '+name+'  AUC= %.2f'%area_cat,)  # linha tracejada azul

    plt.plot( x, baseline_hum,'go') # 
    plt.plot( x, baseline_hum, 'k--', color='green',label='Train Hum Test '+name+' AUC= %.2f'%area_hum)  # linha tracejada azul

    plt.axis([1, 100,0, 1])
    # plt.title("Mais incrementado")
    # plt.rcParams['legend.fontsize'] = 12
    # plt.rc('axes', titlesize=18)
    # plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=18) 
    plt.rcParams.update({'font.size': 13})
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xlabel("Dataset size",size=17)
    plt.ylabel("Precision",size=17)
    plt.savefig('/home/brunafrade/experimentos/evaluate/graficos/'+folder+'/comparative_failure_imageNumber.pdf')
    plt.close()

def graphic_compare_treshold_matplot(ours,baseline_cat,baseline_hum,name,folder):
        x = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
        
        area_ours = trapz(ours, dx=5)
        area_cat = trapz(baseline_cat, dx=5)
        area_hum = trapz(baseline_hum, dx=5)
       
        print len(ours)
        # plt.plot( x, ours, 'go') # green bolinha
        plt.plot( x, ours, 'k:', color='blue',label='Ours  AUC= %.2f'%area_ours) # linha pontilha orange

        plt.plot( x, baseline_cat, 'r^') # red triangulo
        plt.plot( x, baseline_cat, 'k--', color='red',label='Train '+name+' Test '+name+'  AUC= %.2f'%area_cat,)  # linha tracejada azul

        plt.plot( x, baseline_hum,'go') # red triangulo
        plt.plot( x, baseline_hum, 'k--', color='green',label='Train Hum Test '+name+' AUC= %.2f'%area_hum)  # linha tracejada azul

        plt.axis([0, 100,0, 1])
        # plt.title("Mais incrementado")
        # plt.rcParams['legend.fontsize'] = 12

        plt.rcParams.update({'font.size': 13})
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.xlabel("Treshold",size=18)
        plt.ylabel("Precision",size=18)
        plt.rc('axes', titlesize=18)
        plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=18) 
        plt.savefig('/home/brunafrade/experimentos/evaluate/graficos/'+folder+'/comparative_failure_tresh.pdf')
        plt.close()
def graphic_compare_number_image(ours,baseline_cat, baseline_hum,regressor_cat,regressor_hum,name,folder):

    # Add data
    number_image=['0','10','50','100']
    #number_image = ['0','10','25']
    #tif = [32.5, 37.6, 49.9, 53.0, 69.1, 75.4]
    #interspecies = [13.8, 22.3, 32.5, 37.2, 49.9, 56.1]
    # ours = [36.5, 26.6, 43.6, 52.3, 71.5, 81.4]

    area_ours = trapz(ours, dx=5)
    area_cat = trapz(baseline_cat, dx=5)
    area_hum = trapz(baseline_hum, dx=5)

    area_reg_cat = trapz(regressor_cat, dx=5)
    area_reg_hum = trapz(regressor_hum, dx=5)

    # Create and style traces

    trace1 = go.Scatter(
        x = number_image,
        y = ours,
        name = 'Ours A= %.2f'%area_ours+'     ',
        line = dict(
            color = ('rgb(10, 12, 80)'),
            width = 4,
            dash = 'dash') # dash options include 'dash', 'dot', and 'dashdot'
    )
    trace2 = go.Scatter(
        x = number_image,
        y = baseline_cat,
        name = 'Train '+name+' Test '+name+' A= %.2f'%area_cat+'     ',
        line = dict(
            color = ('rgb(205, 12, 24)'),
            width = 4
            ) # dash options include 'dash', 'dot', and 'dashdot'
    )

    trace3 = go.Scatter(
        x = number_image,
        y = baseline_hum,
        name = 'Train Hum Test '+name+' A= %.2f'%area_hum+'     ',
        line = dict(
            color = ('rgb(50, 200, 10)'),
            width = 4
            ) # dash options include 'dash', 'dot', and 'dashdot'
    )

    # trace4 = go.Scatter(
    #     x = number_image,
    #     y = regressor_cat,
    #     name = '(conv 100,200,300,D 1024,6) area: %f'%area_reg_cat,
    #     line = dict(
    #         color = ('rgb(0, 0, 255)'),
    #         width = 4
    #         ) # dash options include 'dash', 'dot', and 'dashdot'
    # )

    # trace5 = go.Scatter(
    #     x = number_image,
    #     y = regressor_hum,
    #     name = 'Regression Layer: Train Hum Test Cat area: %f'%area_reg_hum,
    #     line = dict(
    #         color = ('rgb(0, 0, 0)'),
    #         width = 4
    #         ) # dash options include 'dash', 'dot', and 'dashdot'
    # )



    data = [trace1, trace2, trace3]
    s=23
    # Edit the layout
    layout = dict(legend=dict(
        x=0.5,
        y=1.5,



        font=dict(
            family='sans-serif',
            size = s,
            color='#000'
        ),),
                  xaxis = dict(title = 'Dataset Size (images)',
        font=dict(
            family='sans-serif',
            size = s,
            color='#000'
        ),),
                  yaxis = dict(title = 'Precision %',
        font=dict(
            family='sans-serif',
            size = s,
            color='#000'
        ),),
                  )

    fig = dict(data=data, layout=layout)
    # py.iplot(fig, filename='styled-line')
    fig.savefig(filename='/home/brunafrade/experimentos/evaluate/graficos/'+folder+'/comparative_failure_imageNumber.svg')
    # py.image.save_as(fig, filename='/home/brunafrade/experimentos/evaluate/graficos/'+folder+'/comparative_failure_imageNumber.svg')
def new_accuracy_erickson(keypoint_pred, keypoint_gt,THRESHOLD):
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
    m_orig=np.array([[ 1.,  0.,  0.],[ 0.,  1.,  0.]])


    # Supondo que os pontos estao ordenados em X, Y, X, Y, X, Y...
    for i in range(keypoint_gt.shape[1]/2):
        euc_dist[:, i] = np.sqrt((keypoint_pred[:, 2*i] - keypoint_gt[:, 2*i])**2 + (keypoint_pred[:, 2*i + 1] - keypoint_gt[:, 2*i + 1])**2)
        
        for j in range(keypoint_gt.shape[0]):


            if euc_dist[j, i] > (THRESHOLD * FACE_SIZE) :
                FAILURE_COUNT += 1

    AVG_FAILURE_RATE = float(FAILURE_COUNT) / (keypoint_gt.shape[0] * keypoint_gt.shape[1]/2)

    return AVG_FAILURE_RATE

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
# Load datasets
print('Load datasets')

# Convert class vectors to binary class matrices
nb_classes = 6
animals=['cat','horse']
labels=['Cat','Horse']

cont=0
Xr_train,_, Y_train = load('/home/brunafrade/dataset/training_human_32x32.csv')
for animal in animals:

    Xr_tgt_test,_,y_train_src  = load('/home/brunafrade/dataset/training_'+animal+'_32x32.csv')

    
    # Preprocess input images
    X_train = preprocess_images(Xr_train, tmin=0, tmax=1)
    X_tgt_test = preprocess_images(Xr_tgt_test, tmin=0, tmax=1)
    #param=[0,10,50]
    param=[0,25,10,50,100]
    failure=[]
    failure_ours=[]
    failure_cat=[]

    reg_failure_cat=[]
    reg_failure_hum=[]
    #failure=[4][5]
    for i in param:
        #baseline_hu
        json_file = open('/home/brunafrade/experimentos/evaluate/weights/human/cnn_tresh_{}_conf.json'.format(i), 'r')
        # json_file = open('/home/brunafrade/experimentos/baseline/cat_conf.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        model.load_weights('/home/brunafrade/experimentos/evaluate/weights/human/cnn_tresh_{}_weights.h5'.format(i),by_name=True)
        l=new_accuracy(model.predict(X_tgt_test[:100]),y_train_src[:100],10)
        print l
        failure.append(l)

        # if j==50:
        #     j=25
        #baseline
        json_file = open('/home/brunafrade/experimentos/evaluate/weights/'+animal+'/cnn_tresh_{}_conf.json'.format(i), 'r')
        # json_file = open('/home/brunafrade/experimentos/baseline/'+animal+'_{}_conf.json'.format(j), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        model.load_weights('/home/brunafrade/experimentos/evaluate/weights/'+animal+'/cnn_tresh_{}_weights.h5'.format(i),by_name=True)
        l=new_accuracy(model.predict(X_tgt_test[:100]),y_train_src[:100],10)
        failure_cat.append(l)
        # Y_tgt_test=y_train_src/32.
        
        #ours
        json_file = open('/home/brunafrade/experimentos/evaluate/weights/'+animal+'/ours_tresh_{}_conf.json'.format(i), 'r')
        # json_file = open('/home/brunafrade/experimentos/baseline/cat_conf.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model_ours = model_from_json(loaded_model_json)
        model_ours.load_weights('/home/brunafrade/experimentos/evaluate/weights/'+animal+'/ours_tresh_{}_weights.h5'.format(i),by_name=True)

        l=new_accuracy(model_ours.predict(X_tgt_test[:100]),y_train_src[:100],10)
        failure_ours.append(l)


       



    graphic_compare_number_image_matplot(failure_ours,failure_cat,failure,reg_failure_cat,reg_failure_hum,labels[cont],animal)
    cont+=1



count=0
animals=['cat','dog','horse']
labels=['Cat','Dog','Horse']
for animal in animals:
    Xr_tgt_test,_,y_train_src  = load('/home/brunafrade/dataset/training_'+animal+'_32x32.csv')

    X_tgt_test = preprocess_images(Xr_tgt_test, tmin=0, tmax=1)

    param=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    failure=[]
    failure_ours=[]
    failure_cat=[]

     #baseline
    json_file = open('/home/brunafrade/experimentos/evaluate/weights/human/cnn_tresh_100_conf.json', 'r')
    # json_file = open('/home/brunafrade/experimentos/baseline/cat_conf.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    model.load_weights('/home/brunafrade/experimentos/evaluate/weights/human/cnn_tresh_100_weights.h5',by_name=True)

     #baseline
    json_file = open('/home/brunafrade/experimentos/evaluate/weights/'+animal+'/cnn_tresh_100_conf.json', 'r')
    # json_file = open('/home/brunafrade/experimentos/baseline/cat_conf.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_cat = model_from_json(loaded_model_json)

    model_cat.load_weights('/home/brunafrade/experimentos/evaluate/weights/'+animal+'/cnn_tresh_100_weights.h5',by_name=True)

     #ours
    json_file = open('/home/brunafrade/experimentos/evaluate/weights/'+animal+'/ours_tresh_0_conf.json', 'r')
    # json_file = open('/home/brunafrade/experimentos/baseline/cat_conf.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_ours = model_from_json(loaded_model_json)
    model_ours.load_weights('/home/brunafrade/experimentos/evaluate/weights/'+animal+'/ours_tresh_0_weights.h5',by_name=True)

    for i in param:
       
        l=new_accuracy(model.predict(X_tgt_test[:100]),y_train_src[:100],i-5)
        failure.append(l)

       
        l=new_accuracy(model_cat.predict(X_tgt_test[:100]),y_train_src[:100],i)
        failure_cat.append(l)

        l=new_accuracy(model_ours.predict(X_tgt_test[:100]),y_train_src[:100],i)
        failure_ours.append(l)
    

    graphic_compare_treshold_matplot(failure_ours,failure_cat,failure,labels[count],animal)
    count+=1





# visualization (model_ours, X_tgt_test[:100],X_tgt_test[:100],'','ours_.png')
# visualization (model_cat, X_tgt_test[:100],X_tgt_test[:100],'','base_cat_.png')
# visualization (model, X_tgt_test[:100],X_tgt_test[:100],'','base_hum_.png')