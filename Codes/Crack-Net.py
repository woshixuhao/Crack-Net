# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import torch.utils.data as data
import matplotlib.pyplot as plt
from tqdm import  tqdm
from torch.utils.data import DataLoader
import time
import random
from matplotlib.ticker import MaxNLocator
device = 'cuda' if torch.cuda.is_available() else 'cpu'
v_num=18
use_v_num=3
n=100 #edge length
lc=0.002
use_block=1
(loc_label,loc_E,UTS,loc_gc,loc_coorx,loc_coory,loc_ux,loc_uy,loc_exx,loc_eyy,loc_exy,loc_sxx,loc_syy,loc_szz,loc_sxy,loc_ee0,loc_pe,loc_d)=range(18)
program_name='A5B05'#['PINN-FEM-0515','ML-FEM-05122023']
dataset_name='dataset_'+program_name
def set_seed(seed):
    torch.manual_seed(seed)  # cpu 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed)  # gpu 为当前GPU设置随机种子
    torch.backends.cudnn.deterministic = True  # cudnn
    np.random.seed(seed)  # numpy
    random.seed(seed)

set_seed(1101)

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x

def split_dataset(origin_data):
    all_data = origin_data
    origin_data=origin_data[:,2:]

    data=[]
    for i in range(int(origin_data.shape[1]/v_num)):
        data.append(origin_data[:,0+v_num*i:v_num+v_num*i])
    data=np.array(data)
    global t_num
    t_num=data.shape[1]
    strain = all_data[0:t_num - 1, 0].reshape(-1, 1)
    stress = all_data[0:t_num - 1, 1].reshape(-1, 1)
    stress_predict = all_data[1:t_num, 1].reshape(-1, 1)
    X=data[:,0:t_num-1,:].reshape(n,n,t_num-1,v_num)
    Y=data[:,1:t_num,v_num-1].reshape(n,n,t_num-1,1)
    X=X.transpose(2,3,0,1)
    Y=Y.transpose(2,3,0,1)
    return X,Y,np.hstack([strain,stress,stress_predict])

def save_dataset():
    #-----------构建原始数据集-----------------
    filePath = fr'D:\pycharm project\fanwei\dataset_{program_name}\test'
    file_names=os.listdir(filePath)
    num=0

    for name in file_names:
        a = np.load(f'{dataset_name}/test/'+name)
        X,Y,O=split_dataset(a)
        X=X[:,[1,2,17],:,:]
        np.save(f'{dataset_name}/X_{program_name}_{name}', X)
        np.save(f'{dataset_name}/Y_{program_name}_{name}', Y)
        np.save(f'{dataset_name}/O_{program_name}_{name}', O)

class DownsampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_ReLU_2=nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.downsample=nn.Sequential(
            nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x):
        """
        :param x:
        :return: out输出到深层，out_2输入到下一层，
        """
        out=self.Conv_BN_ReLU_2(x)
        out_2=self.downsample(out)
        return out,out_2

class UpSampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(UpSampleLayer, self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch*2, out_channels=out_ch*2, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU()
        )
        self.upsample_odd=nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch*2,out_channels=out_ch,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch * 2, out_channels=out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self,x,out):
        '''
        :param x: 输入卷积层
        :param out:与上采样层进行cat
        :return:
        '''
        if x.shape[2] in [25,50]:
            x_out = self.Conv_BN_ReLU_2(x)
            x_out = self.upsample_odd(x_out)
        else:
            x_out=self.Conv_BN_ReLU_2(x)
            x_out=self.upsample(x_out)
        cat_out=torch.cat((x_out,out),dim=1)
        return cat_out


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        out_channels=[2**(i+4) for i in range(5)] #[64, 128, 256, 512, 1024]
        #下采样
        self.d1=DownsampleLayer(3,out_channels[0])#3-64
        self.d2=DownsampleLayer(out_channels[0],out_channels[1])#64-128
        self.d3=DownsampleLayer(out_channels[1],out_channels[2])#128-256
        self.d4=DownsampleLayer(out_channels[2],out_channels[3])#256-512
        #上采样
        self.u1=UpSampleLayer(out_channels[3],out_channels[3])#512-1024-512
        self.u2=UpSampleLayer(out_channels[4],out_channels[2])#1024-512-256
        self.u3=UpSampleLayer(out_channels[3],out_channels[1])#512-256-128
        self.u4=UpSampleLayer(out_channels[2],out_channels[0])#256-128-64
        #输出
        self.o=nn.Sequential(
            nn.Conv2d(out_channels[1],out_channels[0],kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(),
            nn.Conv2d(out_channels[0],1,3,1,1),
        )
        self.s_1=nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.fc_SS = nn.Sequential(
            nn.Linear(514, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )


    def forward(self,x,x_SS):
        out_1,out1=self.d1(x)
        out_2,out2=self.d2(out1)
        out_3,out3=self.d3(out2)
        out_4,out4=self.d4(out3)


        out5=self.u1(out4,out_4)
        out6=self.u2(out5,out_3)
        out7=self.u3(out6,out_2)
        out8=self.u4(out7,out_1)
        out=self.o(out8)
        s_4 = self.s_1(out_4)
        s_4=self.flatten(s_4)
        s_4 =  torch.cat((s_4, x_SS), dim=1)
        x_SS = self.fc_SS(s_4)
        return out,x_SS


def plot(X,Y):
    for i in range(X.shape[0]):
        plt.figure(1)
        plt.imshow(X[i,16,:,:])
        plt.colorbar()
        plt.figure(2)
        plt.imshow(Y[i,0,:,:])
        plt.colorbar()
        plt.figure(3)
        plt.imshow(Y[i,0,:,:]-X[i,16,:,:])
        plt.colorbar()
        plt.show()

def save_dataset_shuffle():
    filePath = fr'D:\pycharm project\fanwei\{dataset_name}\train'
    file_names=os.listdir(filePath)
    if len(file_names)<60:
        use_block=1
        split_name=[file_names]
    else:
        split_name= [file_names[i:i+50] for i in range(0,len(file_names),50)]
        use_block = len(split_name)
    print(len(split_name))

    for split_name_index in range(use_block):
        X_total = []
        Y_total = []
        others_total = []
        for name in tqdm(split_name[split_name_index]):
            a = np.load(dataset_name+'/train/'+name)
            X,Y,others=split_dataset(a)
            #============shuffle=============

            X=X[:,[1,2,17],:,:]

            X_total.append(X)
            Y_total.append(Y)
            others_total.append(others)

        X_total=np.vstack(X_total)
        Y_total = np.vstack(Y_total)
        others_total = np.vstack(others_total)
        state = np.random.get_state()
        np.random.shuffle(X_total)
        np.random.set_state(state)
        np.random.shuffle(Y_total)
        np.random.set_state(state)
        np.random.shuffle(others_total)
        np.save(f'{dataset_name}/X_total_{split_name_index}.npy',X_total)
        np.save(f'{dataset_name}/Y_total_{split_name_index}.npy',Y_total)
        np.save(f'{dataset_name}/others_total_{split_name_index}.npy',others_total)
        print(X_total.shape,Y_total.shape,others_total.shape)

def pre_process(X,Y):
    Y=Y-X[:,use_v_num-1,:,:].reshape(X.shape[0],1,X.shape[2],X.shape[3])
    Y[Y <1e-10] = 1e-10
    Y = 10+torch.log10(Y)
    return Y

save_dataset_shuffle()
net=UNet().to(device)

optimizer=torch.optim.Adam([
    {'params': net.parameters()},
])

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, file1_path, target_path,others_path,mode):
        self.file1_path = file1_path
        self.target_path = target_path
        self.others_path=others_path
        self.mode=mode
        self.x1_box = list()
        self.y_box = list()
        self.others_box = list()
        filePath = fr'D:\pycharm project\fanwei\{dataset_name}\train'
        file_names = os.listdir(filePath)
        if len(file_names) < 60:
            use_block=1
        else:
            use_block=int(len(file_names)/50)+1

        for split_num in range(use_block):
            file1_obj = np.load(file1_path + f'_{split_num}' + '.npy')
            target_obj = np.load(target_path + f'_{split_num}' + '.npy')
            others_obj = np.load(others_path + f'_{split_num}' + '.npy')
            train_num = int(0.8 * file1_obj.shape[0])
            val_num = int(0.1 * file1_obj.shape[0])
            test_num = int(0.1 * file1_obj.shape[0])
            if self.mode=='Train':
                file1_obj=file1_obj[0:train_num]
                target_obj = target_obj[0:train_num]
                others_obj = others_obj[0:train_num]
            if self.mode == 'Valid':
                file1_obj = file1_obj[train_num:train_num+val_num]
                target_obj = target_obj[train_num:train_num+val_num]
                others_obj = others_obj[train_num:train_num+val_num]
            if self.mode == 'Test':
                file1_obj = file1_obj[train_num+val_num:train_num+val_num+test_num]
                target_obj = target_obj[train_num+val_num:train_num+val_num+test_num]
                others_obj = others_obj[train_num+val_num:train_num+val_num+test_num]
            for line in range(file1_obj.shape[0]):
                self.x1_box.append(file1_obj[line])
                self.y_box.append(target_obj[line])
                self.others_box.append(others_obj[line])

    def __len__(self):
        return len(self.x1_box)

    def __getitem__(self, index):
        x1 = self.x1_box[index]
        x1 = np.array(x1, dtype='float')
        x1 = torch.from_numpy(x1)
        x1 = x1.type(torch.FloatTensor)


        y = self.y_box[index]
        y = np.array(y, dtype='float')
        y = torch.from_numpy(y)
        y = y.type(torch.FloatTensor)

        others = self.others_box[index]
        others = np.array(others, dtype='float')
        others = torch.from_numpy(others)
        others = others.type(torch.FloatTensor)
        return x1, y,others

X_PATH = f'{dataset_name}/X_total'
Y_PATH = f'{dataset_name}/Y_total'
OTHERS_PATH = f'{dataset_name}/Others_total'
mode='Train'  #['Train','Test','Test_multi']
batch_size_train=100
batch_size_valid=100
batch_size_test=100
if mode=='Train':
    train_dataset = MyDataset(X_PATH,  Y_PATH,OTHERS_PATH,'Train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train)
    validate_dataset = MyDataset(X_PATH, Y_PATH, OTHERS_PATH, 'Valid')
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size_valid)

if mode=='Test':
    test_dataset = MyDataset(X_PATH,  Y_PATH,OTHERS_PATH,'Test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test)


if __name__ == '__main__':
    if mode=='Train':
        model_name = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
        dir_name = 'model_save_'+program_name + '/' + model_name
        loss_plot = []
        loss_validate_plot = []
        try:
            os.makedirs(dir_name)
        except OSError:
            pass

        try:
            os.makedirs(f'fig_save/save_process_{program_name}/')
        except OSError:
            pass

        with open(dir_name + '/' + 'data.txt', 'a+') as f:  # 设置文件对象
            for epoch in tqdm(range(3000)):
                for step, (batch_x, batch_y,batch_other) in enumerate(train_loader):
                    batch_y = pre_process(batch_x, batch_y)
                    optimizer.zero_grad()
                    batch_x = Variable((batch_x).cuda(), requires_grad=True)
                    batch_y = Variable((batch_y).cuda(), requires_grad=True)
                    batch_other = Variable((batch_other).cuda(), requires_grad=True)
                    prediction_origin,predict_SS =net(batch_x,batch_other[:,0:2])
                    prediction = 10 ** (prediction_origin - 10) + batch_x[:, -1, :, :].reshape(batch_x.shape[0], 1,
                                                                                               batch_x.shape[2],
                                                                                               batch_x.shape[3])
                    true_SS = batch_other[:, 2].reshape(-1, 1)
                    MSELoss=torch.nn.MSELoss()
                    data_loss = MSELoss(batch_y, prediction_origin)
                    SS_loss = MSELoss(true_SS, predict_SS)
                    loss = data_loss+0.1*SS_loss
                    loss.backward()
                    optimizer.step()

                if (epoch+1)%5==0:
                    net.eval()

                    for step_valid, (batch_x_valid, batch_y_valid, batch_o_valid) in enumerate(validate_loader):
                        batch_y_valid = pre_process(batch_x_valid, batch_y_valid)
                        batch_x_valid = Variable((batch_x_valid).cuda(), requires_grad=True)
                        batch_y_valid = Variable((batch_y_valid).cuda(), requires_grad=True)
                        batch_o_valid = Variable((batch_o_valid).cuda(), requires_grad=True)
                        prediction_validate, predict_SS_valid = net(batch_x_valid, batch_o_valid[:, 0:2])
                        true_SS_valid = batch_o_valid[:, 2].reshape(-1, 1)
                        loss_validate = MSELoss(batch_y_valid, prediction_validate)
                        loss_validate_SS = MSELoss(true_SS_valid, predict_SS_valid)
                        print("iter_num: %d      loss: %.8f    loss_validate: %.8f    loss_SS: %.8f" % (
                            epoch + 1, loss.item(), loss_validate.item(), loss_validate_SS.item()))
                        break



                    plt.figure(1)
                    plt.subplot(2,2,1)
                    plt.imshow(batch_y_valid[0, 0, :, :].cpu().data.numpy())
                    plt.colorbar()
                    plt.subplot(2, 2, 2)
                    plt.imshow(prediction_validate[0, 0, :, :].cpu().data.numpy())
                    plt.colorbar()
                    plt.subplot(2, 2, 3)
                    plt.imshow(batch_y[0, 0, :, :].cpu().data.numpy())
                    plt.colorbar()
                    plt.subplot(2, 2, 4)
                    plt.imshow(prediction_origin[0, 0, :, :].cpu().data.numpy())
                    plt.colorbar()
                    plt.savefig(f'fig_save/save_process_{program_name}/{epoch}.png')
                    plt.close('all')


                    f.write("iter_num: %d      loss: %.8f    loss_validate: %.8f  loss_SS:  %.8f \r\n" % (
                    epoch + 1, loss.item(), loss_validate.item(),loss_validate_SS.item()))
                    torch.save(net.state_dict(), dir_name + '/' + "net_%d_epoch.pkl" % (epoch+1))
                    torch.save(optimizer.state_dict(),
                               dir_name + '/' + "optimizer_%d_epoch.pkl" % (epoch + 1))
                    loss_plot.append(loss.item())
                    loss_validate_plot.append(loss_validate.item())

                    net.train()



            best_epoch=(loss_validate_plot.index(min(loss_validate_plot))+1)*100
            print("The ANN has been trained, the best epoch is %d"%(best_epoch))
    if mode=='Test':
        dir_name=f'Models'
        try:
            os.makedirs(f'fig_save/test/{program_name}')
        except OSError:
            pass
        try:
            os.makedirs(f'fig_save/train/{program_name}')
        except OSError:
            pass
        net.load_state_dict(torch.load(f'{dir_name}/Crack_net_model.pkl'))
        optimizer.load_state_dict(
            torch.load(f'{dir_name}/Crack_net_optimizer.pkl'))
        net.eval()
        all_SS_true=[]
        all_SS_predict=[]
        all_relative_error_d=[]
        all_relative_error_delta_d = []


        for step, (batch_x, batch_y, batch_o) in enumerate(test_loader):
            batch_y = pre_process(batch_x, batch_y)
            X_test = Variable((batch_x).cuda(), requires_grad=True)
            y_test = Variable((batch_y).cuda(), requires_grad=True)
            O_test = Variable((batch_o).cuda(), requires_grad=True)
            test, predict_SS_test = net(X_test, O_test[:, 0:2])
            true_SS_test = O_test[:, 2].reshape(-1, 1)
            true = y_test.to(device)
            test_diff = 10 ** (test.cpu().data.numpy()[:, 0, :, :] - 10) + X_test.cpu().data.numpy()[:, -1, :, :]
            test_true_diff = 10 ** (true.cpu().data.numpy()[:, 0, :, :] - 10) + X_test.cpu().data.numpy()[:, -1, :, :]
            all_SS_true.extend(true_SS_test.cpu().data.numpy())
            all_SS_predict.extend(predict_SS_test.cpu().data.numpy())
            for i in tqdm(range(batch_x.shape[0])):
                all_relative_error_d.append(np.sqrt(np.sum((test_diff[i] - test_true_diff[i]) ** 2) / np.sum(test_true_diff[i] ** 2)))
                all_relative_error_delta_d.append(np.sqrt(np.sum((test.cpu().data.numpy()[i, 0, :, :] - true.cpu().data.numpy()[i, 0, :, :]) ** 2) / np.sum(true.cpu().data.numpy()[i, 0, :, :] ** 2)))
        all_SS_true=np.array(all_SS_true)
        all_SS_predict=np.array(all_SS_predict)
        plt.style.use('ggplot')
        fig, axes = plt.subplots(1, 1, figsize=(2, 2))
        plt.scatter(all_SS_true, all_SS_predict, color='#8A83B4',s=12,alpha=0.3)
        plt.yticks(fontproperties='Arial', size=7)
        plt.xticks(fontproperties='Arial', size=7)
        plt.plot(np.arange(0,32,1),np.arange(0,32,1),linestyle='--',color='black',linewidth=1.5)
        axes.yaxis.set_major_locator(MaxNLocator(5))
        axes.xaxis.set_major_locator(MaxNLocator(5))
        plt.savefig(f'plot_save/test.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(f'plot_save/test.jpg', bbox_inches='tight', dpi=300)
        plt.show()

        y_true=all_SS_true
        y_pred=all_SS_predict
        R_square = 1 - (((y_true - y_pred) ** 2).sum() / ((y_true - y_pred.mean()) ** 2).sum())
        test_mae = np.mean((y_true - y_pred) ** 2)
        print(R_square,test_mae)
    if mode=='Test_multi':
        save_dataset()
        index=0
        start=0
        for test_name in os.listdir('dataset_0414/test'):
            test_name = test_name[0:-4]
            if index<start:
                index += 1
                continue
            print(test_name)
            version='main_0627'
            more_step=0
            X = np.load(f'{dataset_name}/X_{program_name}_{test_name}.npy')
            X_origin = np.load(f'{dataset_name}/X_{program_name}_{test_name}.npy')
            Y = np.load(f'{dataset_name}/Y_{program_name}_{test_name}.npy')

            O=np.load(f'{dataset_name}/O_{program_name}_{test_name}.npy')
            O_origin = np.load(f'{dataset_name}/O_{program_name}_{test_name}.npy')

            X = torch.from_numpy(X.astype(np.float32))
            Y = torch.from_numpy(Y.astype(np.float32))
            O = torch.from_numpy(O.astype(np.float32))
            X_test = Variable((X).to(device))
            y_test = Variable((Y).to(device))
            y_test = pre_process(X_test, y_test)
            o_test=Variable((O).to(device))

            dir_name=f'Models'
            try:
                os.makedirs(f'fig_save/test_multi/{program_name}/{test_name}/{version}')
            except OSError:
                pass

            net.load_state_dict(torch.load(f'{dir_name}/Crack_net_model.pkl'))
            optimizer.load_state_dict(
                torch.load(f'{dir_name}/Crack_net_optimizer.pkl'))
            net.eval()

            start_index = 0
            test,predict_SS = net(X_test[start_index].unsqueeze(0).to(device),o_test[start_index,0:2].unsqueeze(0).to(device))
            true = y_test[start_index].unsqueeze(0).to(device)
            test_diff = X_test.cpu().data.numpy()[start_index, -1, :, :].reshape([1, 100, 100])
            true_SS = o_test[start_index, 2]
            predict_SS_all = []
            true_SS_all = []
            predict_SS_all.append(predict_SS[0].item())
            true_SS_all.append(true_SS.item())
            for step in tqdm(range(X_test.shape[0] - start_index-1+more_step)):
                test_diff += 10 ** (test.cpu().data.numpy()[:, 0, :, :] - 10)
                if step<X_test.shape[0] - start_index-1:
                    test_true_diff = 10 ** (true.cpu().data.numpy()[:, 0, :, :] - 10) + X_test.cpu().data.numpy()[
                                                                                        start_index + step, -1, :, :]
                test_diff[test_diff > 1] = 1

                for i in range(true.shape[0]):
                    plt.figure(1)
                    plt.subplot(2, 2, 1)
                    plt.imshow(true.cpu().data.numpy()[i, 0, :, :])
                    plt.colorbar()
                    plt.subplot(2, 2, 2)
                    plt.imshow(test.cpu().data.numpy()[i, 0, :, :])
                    plt.colorbar()
                    plt.subplot(2, 2, 3)
                    plt.imshow(test_true_diff[i], vmin=0, vmax=1)
                    plt.colorbar()
                    plt.subplot(2, 2, 4)
                    plt.imshow(test_diff[i], vmin=0, vmax=1)
                    plt.colorbar()
                    plt.savefig(f'fig_save/test_multi/{program_name}/{test_name}/{version}/test_{step}.jpg')
                    plt.clf()

                X_new = X_origin[start_index].reshape(1, X_test.shape[1], X_test.shape[2], X_test.shape[3])
                X_new[:, -1, :, :] = test_diff
                X_test_new = torch.from_numpy(X_new.astype(np.float32))
                X_test_new = Variable((X_test_new).to(device))
                o_new = np.zeros([2, ])
                o_new = torch.from_numpy(o_new.astype(np.float32))
                o_new = Variable((o_new).to(device))
                o_new[0] = O_origin[start_index, 0] + (step + 1) * 0.01
                o_new[1] = predict_SS
                test,predict_SS = net(X_test_new.to(device),o_new.reshape([1,2]).to(device))
                if predict_SS<0:
                    predict_SS*=0

                predict_SS_all.append(predict_SS.item())
                if step < X_test.shape[0] - start_index - 1:
                    true = y_test[start_index + step + 1].reshape(1, y_test.shape[1], y_test.shape[2], y_test.shape[3]).to(
                        device)
                    true_SS = o_test[start_index + step + 1, 2]
                    true_SS_all.append(true_SS.item())
            plt.figure(2)
            plt.plot(np.arange(0, len(predict_SS_all), 1), predict_SS_all, c='red')
            plt.plot(np.arange(0, len(true_SS_all), 1), true_SS_all, c='blue')
            plt.savefig(f'fig_save/test_multi/{program_name}/{test_name}/{version}/SS_curve.jpg')
            plt.clf()
            index+=1


