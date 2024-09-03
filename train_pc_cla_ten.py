import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import torch.nn as nn
import argparse
from model.network import MambaI2P_cla

from datasets.kitti.kitti_pc_img_dataloader import kitti_pc_img_dataset
from options.kitti_options import KITTI_Options
# import loss
from model.loss import  det_loss2, desc_loss
import numpy as np
import logging
import math
import cv2
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import time
import datetime
from torch.utils.tensorboard import SummaryWriter

def get_P_diff(P_pred_np,P_gt_np):
    P_diff=np.dot(np.linalg.inv(P_pred_np),P_gt_np)
    t_diff=np.linalg.norm(P_diff[0:3,3])
    r_diff=P_diff[0:3,0:3]
    R_diff=Rotation.from_matrix(r_diff)
    angles_diff=np.sum(np.abs(R_diff.as_euler('xzy',degrees=True)))
    return t_diff,angles_diff

def model_inference(model,data,opt):
    img_=data['img'].cuda()
    depth_img = data['depth_img'].cuda()                  #full size
    pc=data['pc'].cuda()
    intensity=data['intensity'].cuda()
    sn=data['sn'].cuda()
    K=data['K'].cuda()
    P=data['P'].cuda()
  
    img_mask=data['img_mask'].cuda()        #1/4 size
    pc_kpt_idx=data['pc_kpt_idx'].cuda()    #(B,512)
    pc_outline_idx=data['pc_outline_idx'].cuda()
    img_kpt_idx=data['img_kpt_idx'].cuda()
    img_outline_idx=data['img_outline_index'].cuda()

    img_x=torch.linspace(0,img_mask.size(-1)-1,img_mask.size(-1)).view(1,-1).expand(img_mask.size(-2),img_mask.size(-1)).unsqueeze(0).expand(img_mask.size(0),img_mask.size(-2),img_mask.size(-1)).unsqueeze(1).cuda()
    img_y=torch.linspace(0,img_mask.size(-2)-1,img_mask.size(-2)).view(-1,1).expand(img_mask.size(-2),img_mask.size(-1)).unsqueeze(0).expand(img_mask.size(0),img_mask.size(-2),img_mask.size(-1)).unsqueeze(1).cuda()
    img_xy=torch.cat((img_x,img_y),dim=1)


    #point class inital
    B, img_H, img_W  = img_.size(0), img_.size(2), img_.size(3)
    N = pc.size(2)
    img_W_fine_res = int(round(img_W / opt.img_fine_resolution_scale))
    img_H_fine_res = int(round(img_H / opt.img_fine_resolution_scale))

    
    '''
    点云融合到图像的特征:img_features (B,64,H,W)
    点云融合到图像的相关系数:img_score (B,1,H,W)
    图像融合到点云的特征:pc_features  (B,64,H,W)
    图像融合到点云的相关系数:pc_score  (B,1,H,W)
    '''       
    img = torch.cat((img_, depth_img), dim=1)
    img_siam_feature_norm, pc_siam_feature_norm, \
    img_score,pc_score, pc_class_scores = model(pc,intensity,sn,img)    #64 channels feature

    '''
    选出图像点云的共视区域点云坐标:pc_xyz_inline
    图像点云的共视区域点云i2p特征:pc_features_inline
    图像点云的共视区域点云i2p系数:pc_score_inline
    图像点云的不共视区域点云i2p特征:pc_features_outline
    图像点云的不共视区域点云i2p系数:pc_score_outline
    '''
    
    pc_features_inline=torch.gather(pc_siam_feature_norm,index=pc_kpt_idx.unsqueeze(1).expand(B,pc_siam_feature_norm.size(1),opt.num_kpt),dim=-1)    #(B,C,num_kpt)
    pc_features_outline=torch.gather(pc_siam_feature_norm,index=pc_outline_idx.unsqueeze(1).expand(B,pc_siam_feature_norm.size(1),opt.num_kpt),dim=-1)
    pc_xyz_inline=torch.gather(pc,index=pc_kpt_idx.unsqueeze(1).expand(B,3,opt.num_kpt),dim=-1)
    pc_score_inline=torch.gather(pc_score,index=pc_kpt_idx.unsqueeze(1),dim=-1)         #(B,1,num_in)
    pc_score_outline=torch.gather(pc_score,index=pc_outline_idx.unsqueeze(1),dim=-1)    #(B,1,num_out)

    '''
    选出图像点云的共视区域点图像坐标:img_xy_flatten_inline
    图像点云的共视区域点图像p2i特征:pc_features_inline
    图像点云的共视区域点图像p2i系数:img_score_flatten_inline
    图像点云的不共视区域点云p2i特征:img_features_flatten_outline
    图像点云的不共视区域点云p2i系数:img_score_flatten_outline
    '''              
    # img_features_flatten=img_features.contiguous().view(img_features.size(0),img_features.size(1),-1)   #(B,C,(H*W))

    img_features_flatten = img_siam_feature_norm
    img_score_flatten=img_score.contiguous().view(img_score.size(0),img_score.size(1),-1)               #(B,1,(H*W))
    img_xy_flatten=img_xy.contiguous().view(img_siam_feature_norm.size(0),2,-1)
    img_features_flatten_inline=torch.gather(img_features_flatten,index=img_kpt_idx.unsqueeze(1).expand(B,img_features_flatten.size(1),opt.num_kpt),dim=-1)
    img_xy_flatten_inline=torch.gather(img_xy_flatten,index=img_kpt_idx.unsqueeze(1).expand(B,2,opt.num_kpt),dim=-1)
    img_score_flatten_inline=torch.gather(img_score_flatten,index=img_kpt_idx.unsqueeze(1),dim=-1)
    img_features_flatten_outline=torch.gather(img_features_flatten,index=img_outline_idx.unsqueeze(1).expand(B,img_features_flatten.size(1),opt.num_kpt),dim=-1)
    img_score_flatten_outline=torch.gather(img_score_flatten,index=img_outline_idx.unsqueeze(1),dim=-1)
    

    #----------------------------------------------cal_point_class_loss--------------------------------
    img_W_for_pred = int(img_W * 0.25)
    img_H_for_pred = int(img_H * 0.25)
    # print(img_W_for_pred)
    # print(img_H_for_pred)
    pc_px_py_pz=torch.bmm(K,(torch.bmm(P[:,0:3,0:3],pc)+P[:,0:3,3:]))
    pc_uv = pc_px_py_pz[:,0:2,:]/pc_px_py_pz[:,2:,:]
    x_inside_mask = (pc_uv[:, 0:1, :] >= 0) \
                    & (pc_uv[:, 0:1, :] <= img_W_for_pred - 1)  # Bx1xN
    y_inside_mask = (pc_uv[:, 1:2, :] >= 0) \
                    & (pc_uv[:, 1:2, :] <= img_H_for_pred - 1)  # Bx1xN
    z_inside_mask = pc_px_py_pz[:, 2:3, :] > 0.1  # Bx1xN
    inside_mask = (x_inside_mask & y_inside_mask & z_inside_mask).squeeze(1)  # BxN

    KP_pc_pxpy_scale_int = torch.floor(pc_uv / 8.0).to(dtype=torch.long)
    KP_pc_pxpy_index = KP_pc_pxpy_scale_int[:, 0, :] + KP_pc_pxpy_scale_int[:, 1, :] * int(round(img_W / opt.img_fine_resolution_scale))  # BxN

    # get fine labels
    # organize everything into (B*N)x* shape
    inside_mask_Bn = inside_mask.reshape(B*N)  # BN
    inside_mask_Bn_int = inside_mask_Bn.to(dtype=torch.int32)  # BN
    insider_num = int(torch.sum(inside_mask_Bn_int).item())  # scalar
    _, inside_idx_Bn = torch.sort(inside_mask_Bn_int, descending=True)  # BN
    insider_idx = inside_idx_Bn[0: insider_num]  # B_insider
    
    KP_pc_pxpy_index_Bn = KP_pc_pxpy_index.view(B*N)  # BN in long
    KP_pc_pxpy_index_insider = torch.gather(KP_pc_pxpy_index_Bn, dim=0, index=insider_idx)  # B_insider in long
    # assure correctness
    fine_labels_min = torch.min(KP_pc_pxpy_index_insider).item()
    fine_labels_max = torch.max(KP_pc_pxpy_index_insider).item()
    assert fine_labels_min >= 0
    assert fine_labels_max <= img_W_fine_res * img_H_fine_res - 1
    # BxLxN -> BxNxL
    L = pc_class_scores.size(1)
    fine_scores_BnL = pc_class_scores.permute(0, 2, 1).reshape(B*N, L).contiguous()  # BNxL
    insider_idx_BinsiderL = insider_idx.unsqueeze(1).expand(insider_num, L)  # B_insiderxL
    fine_scores_insider = torch.gather(fine_scores_BnL, dim=0, index=insider_idx_BinsiderL)  # B_insiderxL

    #----------------cal_coview_loss and pixel-point match loss------------------
    #用真实位姿，计算共视区域点云的二维图像投影
    pc_xyz_projection=torch.bmm(K,(torch.bmm(P[:,0:3,0:3],pc_xyz_inline)+P[:,0:3,3:]))
    pc_xy_projection=pc_xyz_projection[:,0:2,:]/pc_xyz_projection[:,2:,:]
    '''
    这有一点类似coarse点对匹配的感觉,计算共视的图像点与点云点,计算两两的重投影误差
    根据重投影误差是否小于opt.dist_thres确定点云-图像相关mask(B,N,N),这步mask的计算使用的全部为真值
    '''
    correspondence_mask=(torch.sqrt(torch.sum(torch.square(img_xy_flatten_inline.unsqueeze(-1)-pc_xy_projection.unsqueeze(-2)),dim=1))<=opt.dist_thres).float()

    #-----------------------cal_loss-----------------------------
    loss_pc_cla = loss_pc_class(fine_scores_insider, KP_pc_pxpy_index_insider)
    #根据features计算loss
    circle_loss,dists = desc_loss(img_features_flatten_inline, pc_features_inline, correspondence_mask,pos_margin=opt.pos_margin,neg_margin=opt.neg_margin)
    #根据相关系数计算loss2
    #就是让inline的(1-score)+outlin的score，当inline_score越大，outline_score越小的时候loss越小
    loss_det=det_loss2(img_score_flatten_inline.squeeze(),img_score_flatten_outline.squeeze(),pc_score_inline.squeeze(),pc_score_outline.squeeze())

    loss_dict = {'loss_pc_cla': loss_pc_cla,
                 'siamese_loss': circle_loss,
                 'loss_det': loss_det}

    #----------------------------------cal_acc----------------------------------------------
    _, fine_predictions_insider = torch.max(fine_scores_insider, dim=1, keepdim=False)
    pc_class_accuracy = torch.sum(torch.eq(KP_pc_pxpy_index_insider, fine_predictions_insider).to(dtype=torch.float)) / insider_num

    pc_scores_flatten = pc_score.permute(0, 2, 1).squeeze(-1).contiguous()  # BNxL
    pc_coview_pre = torch.where(pc_scores_flatten > 0.95, 1.0, 0.0)
    pc_coview_accuracy = torch.sum(torch.eq(inside_mask.to(dtype=torch.long), pc_coview_pre).to(dtype=torch.float)) / ( B * N)

    acc_dict = {'pc_class_accuracy': pc_class_accuracy,
                'pc_coview_accuracy': pc_coview_accuracy}
    return loss_dict, acc_dict


if __name__=='__main__':
    opt=KITTI_Options()

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    logdir=os.path.join(opt.save_path, 'dist_thres_%0.2f_pos_margin_%0.2f_neg_margin_%0.2f'%(opt.dist_thres,opt.pos_margin,opt.neg_margin,))
    try:
        os.makedirs(logdir)
    except:
        print('mkdir failue')

    writer = SummaryWriter()

    train_dataset = kitti_pc_img_dataset(opt.data_path, 'train', opt.input_pt_num,
                                         P_tx_amplitude=opt.P_tx_amplitude,
                                         P_ty_amplitude=opt.P_ty_amplitude,
                                         P_tz_amplitude=opt.P_tz_amplitude,
                                         P_Rx_amplitude=opt.P_Rx_amplitude,
                                         P_Ry_amplitude=opt.P_Ry_amplitude,
                                         P_Rz_amplitude=opt.P_Rz_amplitude,num_kpt=opt.num_kpt,is_front=False)
    test_dataset = kitti_pc_img_dataset(opt.data_path, 'val', opt.input_pt_num,
                                        P_tx_amplitude=opt.P_tx_amplitude,
                                        P_ty_amplitude=opt.P_ty_amplitude,
                                        P_tz_amplitude=opt.P_tz_amplitude,
                                        P_Rx_amplitude=opt.P_Rx_amplitude,
                                        P_Ry_amplitude=opt.P_Ry_amplitude,
                                        P_Rz_amplitude=opt.P_Rz_amplitude,num_kpt=opt.num_kpt,is_front=False)
    assert len(train_dataset) > 10
    assert len(test_dataset) > 10
    trainloader=torch.utils.data.DataLoader(train_dataset,batch_size=opt.train_batch_size,shuffle=True,drop_last=True,num_workers=opt.num_workers)
    testloader=torch.utils.data.DataLoader(test_dataset,batch_size=opt.val_batch_size,shuffle=False,drop_last=True,num_workers=opt.num_workers)

    model=MambaI2P_cla(opt)
    # checkpoints = torch.load('/home/ai-i-sunyunda/code/mambai2p/runs/mamba_cir/dist_thres_1.00_pos_margin_0.20_neg_margin_1.80/base.t7')
    # model.load_state_dict(checkpoints)
    model=model.cuda()

    current_lr=opt.lr
    learnable_params=filter(lambda p:p.requires_grad,model.parameters())
    optimizer=torch.optim.Adam(learnable_params,lr=current_lr)
    # logger.info(opt)

    global_step=0

    best_t_diff=1000
    best_r_diff=1000
    best_test_accuracy = 0

    loss_pc_class = nn.CrossEntropyLoss()
    for epoch in range(opt.epoch):
        
        for step,data in enumerate(trainloader):
        # for step, data in train_loop:
            global_step+=1
            model.train()
            optimizer.zero_grad()
            train_loss_dict, train_acc_dict = model_inference(model, data, opt)

            #Unpack loss_dict
            loss_pc_cla = train_loss_dict['loss_pc_cla']
            siamese_loss = train_loss_dict['siamese_loss']
            loss_det = train_loss_dict['loss_det']

            # loss = siamese_loss * 4  + loss_det * 2 + loss_pc_cla
            loss = siamese_loss * 2  + loss_det * 2 + loss_pc_cla
            loss.backward()
            
            optimizer.step()
            
            #Unpack acc_dict
            pc_class_accuracy = train_acc_dict['pc_class_accuracy']
            pc_coview_accuracy = train_acc_dict['pc_coview_accuracy']

            #----------------------------------------------------------------------------------------

            if global_step%opt.train_batch_size==0:
                logger.info('%s-%d-%d, loss: %f, siamese_loss: %f, loss det: %f, loss_pc_cla: %f, pc_class_acc: %f, pc_coview_acc: %f'%('train',epoch,global_step,loss.data.cpu().numpy(),siamese_loss.data.cpu().numpy(),loss_det.data.cpu().numpy(), loss_pc_cla.data.cpu().numpy(), pc_class_accuracy.data.cpu().numpy(), pc_coview_accuracy.data.cpu().numpy()))

        #epoch done
        test_batch_sum = 0
        test_loss_sum = {'loss_pc_cla': 0,
                        'siamese_loss': 0,
                        'loss_det': 0}
        test_accuracy_sum = {'coview_acc': 0, 'pc_class_acc': 0}
        test_num = 0

        for i, data in enumerate(testloader):
            test_num += 1
            # print(test_num)
            model.eval()
            with torch.no_grad():
                test_loss_dict, test_acc_dict = model_inference(model, data, opt)
            test_batch_sum += opt.val_batch_size
            test_loss_sum['loss_pc_cla'] += opt.val_batch_size * test_loss_dict['loss_pc_cla']
            test_loss_sum['siamese_loss'] += opt.val_batch_size * test_loss_dict['siamese_loss']
            test_loss_sum['loss_det'] += opt.val_batch_size * test_loss_dict['loss_det']
            test_accuracy_sum['coview_acc'] += opt.val_batch_size * test_acc_dict['pc_coview_accuracy']
            test_accuracy_sum['pc_class_acc'] += opt.val_batch_size * test_acc_dict['pc_class_accuracy']
        
        test_loss_sum['loss_pc_cla'] /= test_batch_sum
        test_loss_sum['siamese_loss'] /= test_batch_sum
        test_loss_sum['loss_det'] /= test_batch_sum
        test_accuracy_sum['coview_acc'] /= test_batch_sum
        test_accuracy_sum['pc_class_acc'] /= test_batch_sum
        logger.info('%s-%d, test_siamese_loss: %f, test_loss det: %f, test_loss_pc_cla: %f, test_pc_class_acc: %f, test_pc_coview_acc: %f'%('test',epoch,test_loss_sum['siamese_loss'].data.cpu().numpy(),test_loss_sum['loss_det'].data.cpu().numpy(), test_loss_sum['loss_pc_cla'].data.cpu().numpy(), test_accuracy_sum['pc_class_acc'].data.cpu().numpy(), test_accuracy_sum['coview_acc'].data.cpu().numpy()))

        # record best test loss
        if test_accuracy_sum['pc_class_acc'] > best_test_accuracy:
            best_test_accuracy = test_accuracy_sum['pc_class_acc']
            logger.info('--- best test coarse accuracy %f' % best_test_accuracy)
    
        if epoch%5==0 and epoch>5:
            current_lr=current_lr*0.25
            if current_lr<opt.min_lr:
                current_lr=opt.min_lr
            for param_group in optimizer.param_groups:
                param_group['lr']=current_lr
            logger.info('%s-%d-%d, updata lr, current lr is %f'%('train',epoch,global_step,current_lr))
        torch.save(model.state_dict(),os.path.join(logdir,'mode_epoch_%d.t7'%epoch))