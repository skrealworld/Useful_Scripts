"""
Script:


"""

import scipy.io as sio 
import numpy as np

start_vid = 21
end_vid = 70
frames_group_num = 500
X_train = np.random.rand(1,500,2048)

for i in range(start_vid,end_vid):
    """

    """
    vid_data = sio.loadmat('./resNet/' + str(i) + '.mat')
    vid_data = vid_data['x']
    vid_data = np.transpose(vid_data) 
    num_frames = vid_data.shape[0]
    print ("Number of frames in video " + str(i) + " is " +  str(num_frames))
    #print(vid_data.shape)
    vid_idx = num_frames/frames_group_num 
    append_num = (vid_idx+1)*500-num_frames
    append_arr = np.zeros((append_num,2048),dtype=float )
    vid_data = np.concatenate((vid_data,append_arr),axis=0)
    #print('AFTER APPENDIGN')
    #print(vid_data.shape)
"""
    for j in range(0,2*vid_idx+1):
       
        if j==0&i==start_vid:
            X_train[0] = vid_data[0:500,:]
        else:
            temp_data = np.random.rand(1,500,2048)
            temp_data[0] = vid_data[j*250:(j+2)*250,:]
            X_train = np.concatenate((X_train,temp_data),axis=0)
            

"""                
    
