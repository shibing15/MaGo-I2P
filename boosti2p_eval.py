import numpy as np
import cv2
import os
import torch
from model.network import MambaI2P_cla
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from scipy.spatial.transform import Rotation
import argparse
from datasets.kitti.kitti_pc_img_dataloader import kitti_pc_img_dataset
from datasets.oxford.siami2p_oxford_pc_img_dataloader import OxfordLoader
from evalution.model_eval import Model_eval
from options.kitti_options import KITTI_Options
import math
import sys
from tqdm import tqdm
import time

def get_P_diff(P_pred_np,P_gt_np):
    P_diff=np.dot(np.linalg.inv(P_pred_np),P_gt_np)
    t_diff=np.linalg.norm(P_diff[0:3,3])
    r_diff=P_diff[0:3,0:3]
    R_diff=Rotation.from_matrix(r_diff)
    angles_diff=np.sum(np.abs(R_diff.as_euler('xzy',degrees=True)))

    return t_diff,angles_diff


def model_inference(model,data,opt):

    #----------------------------------Unpack data------------------------------------------------------
    img_=data['img'].cuda()
    depth_img = data['depth_img'].cuda()                  #full size
    pc=data['pc'].cuda()
    intensity=data['intensity'].cuda()
    sn=data['sn'].cuda()
    img_mask=data['img_mask'].cuda()        #1/4 size
    img_x=torch.linspace(0,img_mask.size(-1)-1,img_mask.size(-1)).view(1,-1).expand(img_mask.size(-2),img_mask.size(-1)).unsqueeze(0).expand(img_mask.size(0),img_mask.size(-2),img_mask.size(-1)).unsqueeze(1).cuda()
    img_y=torch.linspace(0,img_mask.size(-2)-1,img_mask.size(-2)).view(-1,1).expand(img_mask.size(-2),img_mask.size(-1)).unsqueeze(0).expand(img_mask.size(0),img_mask.size(-2),img_mask.size(-1)).unsqueeze(1).cuda()
    img_xy=torch.cat((img_x,img_y),dim=1)

    #-----------------------------------param for accuracy teset---------------------------------
    K=data['K'].cuda()
    P=data['P'].cuda()
    B, img_H, img_W  = img_.size(0), img_.size(2), img_.size(3)
    N = pc.size(2)
    img_W_fine_res = int(round(img_W / opt.img_fine_resolution_scale))
    img_H_fine_res = int(round(img_H / opt.img_fine_resolution_scale))

    img_W_for_pred = int(round(img_W / 4.0))
    img_H_for_pred = int(round(img_H / 4.0))

    output_param_dict = {'K': K, 'P': P, 'B': B, 'N':N,
                    'img_W': img_W, 'img_H':img_H,
                    'img_W_fine_res':img_W_fine_res,
                    'img_H_fine_res':img_H_fine_res,
                    'img_W_for_pred' : img_W_for_pred,
                    'img_H_for_pred' : img_H_for_pred,
                    'img_fine_resolution_scale': opt.img_fine_resolution_scale}
    
    #-----------------------------------model ouput----------------------------------------------
    '''
    点云融合到图像的特征:img_features (B,64,H,W)
    点云融合到图像的相关系数:img_score (B,1,H,W)
    图像融合到点云的特征:pc_features  (B,64,H,W)
    图像融合到点云的相关系数:pc_score  (B,1,H,W)
    '''       
    img = torch.cat((img_, depth_img), dim=1)
    img_siam_feature_norm, pc_siam_feature_norm, \
    img_score,pc_score = model(pc,intensity,sn,img)    #64 channels feature
    
    model_output_dict={'img_siam_feature_norm':img_siam_feature_norm,
                       'pc_siam_feature_norm':pc_siam_feature_norm,
                        'img_score':img_score, 'pc_score':pc_score}

    #--------------------------------cal labels-------------------------------------------------

    #---------get coview labels
    pc_px_py_pz=torch.bmm(K,(torch.bmm(P[:,0:3,0:3],pc)+P[:,0:3,3:]))
    pc_uv = pc_px_py_pz[:,0:2,:]/pc_px_py_pz[:,2:,:]
    x_inside_mask = (pc_uv[:, 0:1, :] >= 0) \
                    & (pc_uv[:, 0:1, :] <= img_W_for_pred - 1)  # Bx1xN
    y_inside_mask = (pc_uv[:, 1:2, :] >= 0) \
                    & (pc_uv[:, 1:2, :] <= img_H_for_pred - 1)  # Bx1xN
    z_inside_mask = pc_px_py_pz[:, 2:3, :] > 0.1  # Bx1xN
    
    #coview labels
    inside_mask = (x_inside_mask & y_inside_mask & z_inside_mask).squeeze(1)  # BxN

    #----------cal class for each point
    KP_pc_pxpy_scale_int = torch.floor(pc_uv / 8.0).to(dtype=torch.long)
    KP_pc_pxpy_index = KP_pc_pxpy_scale_int[:, 0, :] + KP_pc_pxpy_scale_int[:, 1, :] * int(round(img_W / opt.img_fine_resolution_scale))  # BxN

    #organize everything into (B*N)x* shape
    inside_mask_Bn = inside_mask.reshape(B*N)  # BN
    inside_mask_Bn_int = inside_mask_Bn.to(dtype=torch.int32)  # BN
    insider_num = int(torch.sum(inside_mask_Bn_int).item())  # scalar int
    _, inside_idx_Bn = torch.sort(inside_mask_Bn_int, descending=True)  # BN
    insider_idx = inside_idx_Bn[0: insider_num]  # B_insider (B,)
    KP_pc_pxpy_index_Bn = KP_pc_pxpy_index.view(B*N)  # BN in long
    

    #pc_calss labels
    KP_pc_pxpy_index_insider = torch.gather(KP_pc_pxpy_index_Bn, dim=0, index=insider_idx)  # B_insider in long (B*insider_num,)

    #----------cal pixel corrdinate for each point
    KP_pc_pxpy_int = torch.round(pc_uv)   #(B,2,N)
    KP_pc_pxpy_index_pixel = KP_pc_pxpy_int[:, 0, :] + KP_pc_pxpy_int[:, 1, :] * int(round(img_W_for_pred))
    KP_pc_pxpy_index_pixel_Bn = KP_pc_pxpy_index_pixel.view(B*N)

    #pixel-point match labels
    KP_pc_pxpy_index_pixel_insider = torch.gather(KP_pc_pxpy_index_pixel_Bn, dim=0, index=insider_idx) # (B*insider_num,)

    # assure correctness
    fine_labels_min = torch.min(KP_pc_pxpy_index_insider).item()
    fine_labels_max = torch.max(KP_pc_pxpy_index_insider).item()
    assert fine_labels_min >= 0
    assert fine_labels_max <= img_W_fine_res * img_H_fine_res - 1

    #-------------------------cal pixel-point match acc--------------------------------------
    
    #get inside points and pixels index 
    pc_score_Bn = pc_score.reshape(B*N)
    pc_score_inside_index_pred = torch.where(pc_score_Bn > 0.95)
    
    img_score_Bn = img_score.reshape((B , int(img_W * 0.25) , int(img_H * 0.25)))
    img_score_inside_index_pred = torch.where(img_score_Bn > 0.95)

    #for each pc class, get the inside points index belong to the class, use dict?
    range_num = max(pc_score_inside_index_pred[0].size(), img_score_inside_index_pred[0].size())
    
    #for each pc class, get the inside pixels index belong to the class, use dict?
    
    #from above two dict, cal the simliarites between pixel and point and build the 2d-3d match index, use list?


    label_dict = {'insider_num': insider_num,
                'insider_idx': insider_idx,
                'coview_labels':inside_mask, 
                'pc_class_labels': KP_pc_pxpy_index_insider,
                'pc_pixel_labels': KP_pc_pxpy_index_pixel_Bn}
    # print('insider_num: ' + str(insider_num) + '\n'
    #     'insider_idx: ' + str(insider_idx.size()) + '\n'
    #     'coview_labels: ' + str(inside_mask.size()) + '\n'
    #     'pc_class_labels: ' + str(KP_pc_pxpy_index_insider.size()) + '\n'
    #     'KP_pc_pxpy_int: ' + str(KP_pc_pxpy_int.size()) + '\n'
    #     'pixel_point_labels: ' + str(KP_pc_pxpy_index_pixel_insider.size()))
    # print('img_siam_feature_norm: ' + str(img_siam_feature_norm.size()) + '\n'
    #     'pc_siam_feature_norm: ' + str(pc_siam_feature_norm.size()) + '\n'
    #     'img_score: ' + str(img_score.size()) + '\n'
    #     'pc_score: ' + str(pc_score.size()) + '\n'
    #     'pc_class_scores: ' + str(pc_class_scores.size()))

    return output_param_dict, model_output_dict, label_dict

def cal_acc(output_param_dict, model_output_dict, label_dict, data, t_error_list, r_error_list, dist):
    #----------------------------------------------Unpack input data--------------------------------
    
    B = output_param_dict['B']
    N = output_param_dict['N']
    img_W = output_param_dict['img_W']
    img_H = output_param_dict['img_H']
    img_W_fine_res = output_param_dict['img_W_fine_res']
    img_H_fine_res = output_param_dict['img_H_fine_res']
    img_fine_resolution_scale = output_param_dict['img_fine_resolution_scale']

    pc_score = model_output_dict['pc_score']
    img_score = model_output_dict['img_score']
    img_siam_feature_norm = model_output_dict['img_siam_feature_norm']
    pc_siam_feature_norm = model_output_dict['pc_siam_feature_norm']

    coview_labels = label_dict['coview_labels']
    pc_class_labels = label_dict['pc_class_labels']
    insider_num = label_dict['insider_num']
    insider_idx = label_dict['insider_idx']
    pc_pixel_labels = label_dict['pc_pixel_labels']

    #--------------------------------cal pc co_view acc------------------------------------------------
    pc = data['pc'].cuda()
    img_W_for_pred = int(img_W * 0.25)
    img_H_for_pred = int(img_H * 0.25)

    pc_scores_flatten = pc_score.permute(0, 2, 1).squeeze(-1).contiguous()  # BNxL
    pc_coview_pre = torch.where(pc_scores_flatten > 0.95, 1.0, 0.0)
    pc_coview_accuracy = torch.sum(torch.eq(coview_labels.to(dtype=torch.long), pc_coview_pre).to(dtype=torch.float)) / ( B * N)
    
    #selecting features in co-view region
    
    pc_scores_Bn = pc_scores_flatten.reshape(B*N)
    pc_inside_flatten = torch.where(pc_scores_Bn > 0.95)[0]

    pc_inside_pred = pc[:, :, pc_inside_flatten].squeeze(0)
    pc_feat_in = pc_siam_feature_norm[:, :, pc_inside_flatten].squeeze(0)
    pc_pixel_label_inside_pred = pc_pixel_labels[pc_inside_flatten].squeeze(0)

    img_x_cla_label=torch.linspace(0,img_W_for_pred-1,img_W_for_pred).reshape(1,-1).repeat(img_H_for_pred,1).reshape(1,img_H_for_pred,img_W_for_pred).cuda()
    img_y_cla_label=torch.linspace(0,img_H_for_pred-1,img_H_for_pred).reshape(-1,1).repeat(1, img_W_for_pred).reshape(1,img_H_for_pred,img_W_for_pred).cuda()
    img_xy_flatten = (img_y_cla_label * img_W_for_pred  + img_x_cla_label).reshape(B*img_W_for_pred*img_H_for_pred)
    img_score_flatten = img_score.reshape(B * img_W_for_pred * img_H_for_pred)
    img_inside_index = torch.where(img_score_flatten > 0.95)[0]

    pixel_in = img_xy_flatten[img_inside_index]
    pixel_feat_in = img_siam_feature_norm[:, :, img_inside_index].squeeze(0)
    #---------------------------------- pred pose---------------------------------------------
    correlation_map = 1 - torch.mm(pixel_feat_in.t(), pc_feat_in)
    pixel_match_pc_index = torch.argmin(correlation_map, dim=1)

    match_dict_pc = {str(key) :[] for key in range(img_W_fine_res* img_H_fine_res) }
    match_dict_pixel = {str(key) :[] for key in range(img_W_fine_res* img_H_fine_res) }

    

    #---------------------------------- pred pose---------------------------------------------
    pc_matched_np = pc_inside_pred[:, pixel_match_pc_index].data.cpu().numpy()
    pixel_matched_y = torch.floor(pixel_in / img_W_for_pred)
    pixel_matched_x = pixel_in - pixel_matched_y * img_W_for_pred
    pixel_matched_xy = torch.cat([pixel_matched_x.unsqueeze(0), pixel_matched_y.unsqueeze(0)], dim=0)
    pixel_matched_xy_np = pixel_matched_xy.data.cpu().numpy()
    K_np = data['K'].squeeze(0).data.cpu().numpy()
    P_np = data['P'].squeeze(0).data.cpu().numpy()
    try:
        is_success,R,t,inliers=cv2.solvePnPRansac(pc_matched_np.T,pixel_matched_xy_np.T,K_np,useExtrinsicGuess=False,
                                                    iterationsCount=500,
                                                    reprojectionError=1,
                                                    flags=cv2.SOLVEPNP_EPNP,
                                                    distCoeffs=None)
    except:
        # print(num*img_score_set.shape[0]+i,'has problem!')
        print('pc shape',pc_matched_np.shape,'img shape',pixel_matched_xy_np.shape)
        assert False
    R,_=cv2.Rodrigues(R)
    T_pred=np.eye(4)
    T_pred[0:3,0:3]=R
    T_pred[0:3,3:]=t
    t_diff,angles_diff=get_P_diff(T_pred,P_np)
    if(is_success):
        t_error_list.append(t_diff)
        r_error_list.append(angles_diff)
    # print(t_diff,angles_diff)

    #根据pc_match_index_list, pixel_match_index_list,计算point-pixel匹配正确率,需要pc及对应的像素坐标真值
    # pc = data['pc']
    # img = data[]



    acc_dict = {'pc_coview_accuracy': pc_coview_accuracy}

    return acc_dict


def test_model(model_path, model, test_dataloader, opt):
    t_error_list = []
    r_error_list = []
    pc_coview_accuracy_list = []
    pc_class_accuracy_list = []
    mask_list = []
    checkpoints = torch.load(model_path)
    model.load_state_dict(checkpoints)
    model = model.cuda()
    with torch.no_grad():
        for data in tqdm(test_dataloader):
            model.eval()
            model_eval = Model_eval(model, data, opt)

            pc_coview_accuracy,\
            pc_class_accuracy,\
            t_diff, angles_diff = model_eval.forward()
            
            # print(t_diff, angles_diff)

            t_error_list.append(t_diff)
            r_error_list.append(angles_diff)
            pc_coview_accuracy_list.append(pc_coview_accuracy)
            pc_class_accuracy_list.append(pc_class_accuracy)
            
    return t_error_list, r_error_list, pc_coview_accuracy_list, pc_class_accuracy_list




if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--val_batch_size', type=int, default=1, metavar='val_batch_size',
                        help='Size of val batch')
    parser.add_argument('--dataset', type=str, default='kitti', metavar='dataset')
    parser.add_argument('--data_path', type=str, default='/home/data/syd/KITTI_for_DEEPI2P', metavar='data_path',
                        help='train and test data path')
    parser.add_argument('--num_point', type=int, default=40960, metavar='num_point',
                        help='point cloud size to train')
    parser.add_argument('--num_workers', type=int, default=8, metavar='num_workers',
                        help='num of CPUs')
    parser.add_argument('--P_tx_amplitude', type=float, default=10, metavar='P_tx_amplitude',
                        help='')
    parser.add_argument('--P_ty_amplitude', type=float, default=0, metavar='P_ty_amplitude',
                        help='')
    parser.add_argument('--P_tz_amplitude', type=float, default=10, metavar='P_tz_amplitude',
                        help='')
    parser.add_argument('--P_Rx_amplitude', type=float, default=2*math.pi*0, metavar='P_Rx_amplitude',
                        help='')
    parser.add_argument('--P_Ry_amplitude', type=float, default=2*math.pi, metavar='P_Ry_amplitude',
                        help='')
    parser.add_argument('--P_Rz_amplitude', type=float, default=2*math.pi*0, metavar='P_Rz_amplitude',
                        help='')    
    parser.add_argument('--num_kpt', type=int, default=512, metavar='num_kpt',
                        help='')
    parser.add_argument('--img_thres', type=float, default=0.95, metavar='img_thres',
                        help='')
    parser.add_argument('--pc_thres', type=float, default=0.95, metavar='pc_thres',
                        help='')
    parser.add_argument('--test_model_path', type=str, default=None, metavar='test_model_path',
                        help='')
    args = parser.parse_args()
    
    if args.dataset == 'kitti':
        test_dataset = kitti_pc_img_dataset(args.data_path, 'val', args.num_point,
                                        P_tx_amplitude=args.P_tx_amplitude,
                                        P_ty_amplitude=args.P_ty_amplitude,
                                        P_tz_amplitude=args.P_tz_amplitude,
                                        P_Rx_amplitude=args.P_Rx_amplitude,
                                        P_Ry_amplitude=args.P_Ry_amplitude,
                                        P_Rz_amplitude=args.P_Rz_amplitude,num_kpt=args.num_kpt,is_front=False)
        opt=KITTI_Options()
        print('load kitti options')
    elif args.dataset == 'oxford':
        # opt = ox_Options()
        # test_dataset = OxfordLoader(args.data_path, 'val', opt=opt)
        print('load oxoford options')
    else:
        print('please input dataset')
        sys.exit()
    
    assert len(test_dataset) > 10
    testloader=torch.utils.data.DataLoader(test_dataset,batch_size=args.val_batch_size,shuffle=False,drop_last=True,num_workers=args.num_workers)
    
    model=MambaI2P_cla(opt)
    model = model.cuda()
    modellist = []
    if args.test_model_path != None:
        filelist = os.listdir(args.test_model_path)
        for file in filelist:
            folder_path = os.path.join(args.test_model_path, file)
            if (not os.path.isdir(folder_path)) and file.endswith('.t7'):
                filesplit = file.split('_')
                if len(filesplit) > 0:
                    modellist.append(file)
        print(modellist, ' will be test')
        for checkpoint in modellist:
            save_add = checkpoint.split('.')
            if os.path.exists(os.path.join(args.test_model_path,save_add[0] + '_t_error' + '.npy')):
                print(checkpoint, ' done!')
                continue
            print(os.path.join(args.test_model_path,save_add[0] + '_t_error_' + '.npy'))
            print(checkpoint, ' test start!!')
            checkpoint_path = os.path.join(args.test_model_path, checkpoint)

            t_error_list, \
            r_error_list, \
            pc_coview_accuracy_list, \
            pc_class_accuracy_list = test_model(checkpoint_path, model, testloader, opt)

            np.save(os.path.join(args.test_model_path,save_add[0] + '_t_error'+'.npy'), t_error_list)
            np.save(os.path.join(args.test_model_path,save_add[0] + '_r_error'+'.npy'), r_error_list)
            np.save(os.path.join(args.test_model_path,save_add[0] + '_coview_acc'+'.npy'),pc_coview_accuracy_list)
            np.save(os.path.join(args.test_model_path,save_add[0] + '_cla_acc'+'.npy'), pc_class_accuracy_list)
            print(checkpoint, ' done!')
            #check if there are any new models have done
            filelist = os.listdir(args.test_model_path)
            for file in filelist:
                folder_path = os.path.join(args.test_model_path, file)
                if (not os.path.isdir(folder_path)) and file.endswith('.t7'):
                    filesplit = file.split('_')
                    if len(filesplit) > 0:
                        modellist.append(file)
    else:
        print('No model input could be eval, please check args.test_model or args.test_model_path')

