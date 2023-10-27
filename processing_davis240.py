import os
import scipy.io as scio
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default=True, help="load data path")
parser.add_argument("--path1", type=str, default=True, help="save path of vol file")
parser.add_argument("--path2", type=str, default=True, help="save path of npy file")
args = parser.parse_args()


def e2v(fea, events,h,w,  bins=24, c=1):

    img_fea = torch.zeros(c, bins * h * w)
    
    feai = fea
    t_ind =(events[:,-1] - events[:,-1].min()) / (events[:,-1].max() - events[:,-1].min() + 1e-5) 
    t_ind = np.clip(np.array(t_ind * bins,dtype=np.int32),0,bins-1)

    x_ind = np.array(events[:,0],dtype=np.int32)
    y_ind = np.array(events[:,1],dtype=np.int32)



    ind = (h * w * t_ind + w * y_ind + x_ind) 
  
    img_fea.index_add_( 1, torch.tensor(ind), torch.tensor(events[:,-2]).reshape(1,-1) )


    final = img_fea.reshape(c,bins,h,w)
    
    return final


def process(root,savepath,savenpypath):
    timescale = 1e6
    t_shift = -0.04
    eventslength_list = []
    eventsstamps_list = []
    ratio = []
    # savepath=r'E:\deblurdata\Real_Dataset\volfile'
    # savenpypath=r"E:\deblurdata\Real_Dataset\npyfile"
    for file in os.listdir(root):
      
        subfile =os.path.join(root,file,'data.mat')
        blurfile= os.path.join(root,file,'blurimage')

        print("--------------------processing ",subfile," --------------------------")
        framelen= len(os.listdir(blurfile))
        data=scio.loadmat(subfile)
        y_o = data['matlabdata'][0,0]['data'][0,0]['polarity'][0,0]['y']
        x_o = data['matlabdata'][0,0]['data'][0,0]['polarity'][0,0]['x']  
        pol_o = data['matlabdata'][0,0]['data'][0,0]['polarity'][0,0]['polarity']  
        t_o = data['matlabdata'][0,0]['data'][0,0]['polarity'][0,0]['timeStamp']

        y_o = np.array(y_o,dtype=np.float32)-1
        x_o = np.array(x_o,dtype=np.float32)-1
        pol_o = np.array(pol_o,dtype=np.float32)
        pol_o[pol_o==0] = -1
        t_o = np.array(t_o,dtype=np.float32)/timescale
        events = np.hstack([x_o,y_o,pol_o,t_o])

        
        for frame in range(1,framelen-1):
            t_for = np.array(data['matlabdata'][0,0]['data'][0,0]['frame'][0,0]['timeStampStart'][frame+1],dtype=np.float32)/timescale -\
                    np.array(data['matlabdata'][0,0]['data'][0,0]['frame'][0,0]['timeStampEnd'][frame],dtype=np.float32)/timescale 
            t_back = np.array(data['matlabdata'][0,0]['data'][0,0]['frame'][0,0]['timeStampStart'][frame],dtype=np.float32)/timescale -\
                    np.array(data['matlabdata'][0,0]['data'][0,0]['frame'][0,0]['timeStampEnd'][frame-1],dtype=np.float32)/timescale 
            
            eventstart = np.array(data['matlabdata'][0,0]['data'][0,0]['frame'][0,0]['timeStampStart'][frame],dtype=np.float32)/timescale + t_shift - t_back/2
            eventend =  np.array(data['matlabdata'][0,0]['data'][0,0]['frame'][0,0]['timeStampEnd'][frame],dtype=np.float32)/timescale + t_shift + t_for/2

            EVENT = events[(events[:,-1]>=eventstart) & (events[:,-1]<=eventend)]
            np.save(savenpypath+"/"+file+str(frame),EVENT) 



            stamps = len(np.unique(EVENT[:,3]))
            vol= e2v(events,events,h=180,w=240,bins=stamps//100)
            ratio.append((vol==0).sum()/vol.shape[1]/vol.shape[2]/vol.shape[3])
            np.save(savepath+"/"+file+str(frame),vol)  
            eventslength_list.append(len(EVENT))
            eventsstamps_list.append(stamps)
       
    print("mean eventslength:",np.mean(eventslength_list))
    print("mean eventstamps:",np.mean(eventsstamps_list))
    print("mean ratio:",np.mean(ratio))

if __name__=="__main__":

    # root= 'E:\deblurdata\Real_Dataset\matfile'
    process(args.root,args.path1,args.path2)