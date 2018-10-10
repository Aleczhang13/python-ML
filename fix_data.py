import os
import scipy.io as scio
import h5py
import numpy as np
def return_file_name(path):
    return  os.listdir(path)

if __name__ =="__main__":
    path = "./dataset/COL20"
    path_list = return_file_name(path)
    X_DATA = []
    Y_DATA = []
    for file in path_list:
        file_path = path + "/" + file
        content = scio.loadmat(file_path)
        X_src,Y_src,X_tar,Y_tar = content['X_src'],content["Y_src"],content["X_tar"],content["Y_tar"]
        ## 转化为四维图像数据
        picture_list = []
        for i in range(X_src.shape[1]):
            tmp = X_src[:,i].reshape(32,32)
            tmp = tmp[:,:, np.newaxis]
            tmp = np.expand_dims(tmp, axis=0)
            picture_list.append(tmp)
        for i in range(X_tar.shape[1]):
            tmp = X_tar[:, i].reshape(32, 32)
            tmp = tmp[:, :, np.newaxis]
            tmp = np.expand_dims(tmp, axis=0)
            picture_list.append(tmp)

        X = np.concatenate([x for x in picture_list])
        X_DATA.append(X)

        ## 第一列是class，整数形式；第二列是domain，整数形式。
        domin_src = np.zeros(Y_src.shape,dtype=int)
        domin_tar = np.ones(Y_tar.shape,dtype=int)
        Y_1 = np.concatenate([Y_src,domin_src],axis=1)
        Y_2 = np.concatenate([Y_tar,domin_tar],axis=1)
        Y = np.concatenate([Y_1,Y_2],axis=0)
        Y_DATA.append(Y)

        # 创造h5py
    f = h5py.File('COIL_2.h5', 'w')
    f.create_dataset('X', data=X_DATA[0])
    f.create_dataset('Y', data=Y_DATA[0])
    f.close()

    f_1 = h5py.File("COIL_1.h5", "w")
    f_1.create_dataset('X', data=X_DATA[1])
    f_1.create_dataset('Y', data=Y_DATA[1])
    f_1.close()
    # #
    # f = h5py.File('COIL_2.h5', 'r')
    # X = f['X']
    # Y = f['Y']
    # print(X.shape)
    # f.close()



















