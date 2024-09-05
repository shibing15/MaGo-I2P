import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import torch
import time

def get_P_diff(P_pred_np,P_gt_np):
    P_diff=np.dot(np.linalg.inv(P_pred_np),P_gt_np)
    t_diff=np.linalg.norm(P_diff[0:3,3])
    r_diff=P_diff[0:3,0:3]
    R_diff=Rotation.from_matrix(r_diff)
    angles_diff=np.sum(np.abs(R_diff.as_euler('xzy',degrees=True)))

    return t_diff,angles_diff


class Model_eval:
    def __init__(self, model, data, opt):
        #-------------unpack data------------------
        self.img_=data['img'].cuda()
        self.depth_img = data['depth_img'].cuda()                  #full size
        self.pc=data['pc'].cuda()
        self.intensity=data['intensity'].cuda()
        self.sn=data['sn'].cuda()
        self.img_mask=data['img_mask'].cuda()        #1/4 size
        self.K=data['K'].cuda()
        self.P=data['P'].cuda()

        self.B, self.img_H, self.img_W  = self.img_.size(0), self.img_.size(2), self.img_.size(3)
        self.N = self.pc.size(2)
        self.img_fine_resolution_scale = opt.img_fine_resolution_scale
        self.img_W_fine_res = int(round(self.img_W / self.img_fine_resolution_scale))
        self.img_H_fine_res = int(round(self.img_H / self.img_fine_resolution_scale))

        self.img_W_for_pred = int(round(self.img_W / 4.0))
        self.img_H_for_pred = int(round(self.img_H / 4.0))

        #generate pixel_seq
        img_x=torch.linspace(0,self.img_mask.size(-1)-1,self.img_mask.size(-1)).view(1,-1).expand(self.img_mask.size(-2),self.img_mask.size(-1)).unsqueeze(0).expand(self.img_mask.size(0),self.img_mask.size(-2),self.img_mask.size(-1)).unsqueeze(1)
        img_y=torch.linspace(0,self.img_mask.size(-2)-1,self.img_mask.size(-2)).view(-1,1).expand(self.img_mask.size(-2),self.img_mask.size(-1)).unsqueeze(0).expand(self.img_mask.size(0),self.img_mask.size(-2),self.img_mask.size(-1)).unsqueeze(1)
        self.img_xy=torch.cat((img_x, img_y),dim=1)

        #init model
        self.model = model

        img = torch.cat((self.img_, self.depth_img), dim=1)
        '''
        点云融合到图像的特征:img_features (B,64,H,W)
        点云融合到图像的相关系数:img_score (B,1,H,W)
        图像融合到点云的特征:pc_features  (B,64,H,W)
        图像融合到点云的相关系数:pc_score  (B,1,H,W)
        '''     
        self.img_siam_feature_norm, \
        self.pc_siam_feature_norm, \
        self.img_score,\
        self.pc_score, \
        self.pc_class_scores = self.model(self.pc,self.intensity,self.sn,img)    #64 
        self.L = self.pc_class_scores.size(1)
        
    def cal_labels_(self):
        #---------get coview labels
        pc_px_py_pz=torch.bmm(self.K,(torch.bmm(self.P[:,0:3,0:3],self.pc)+self.P[:,0:3,3:]))
        pc_uv = pc_px_py_pz[:,0:2,:]/pc_px_py_pz[:,2:,:]
        x_inside_mask = (pc_uv[:, 0:1, :] >= 0) \
                        & (pc_uv[:, 0:1, :] <= self.img_W_for_pred - 1)  # Bx1xN
        y_inside_mask = (pc_uv[:, 1:2, :] >= 0) \
                        & (pc_uv[:, 1:2, :] <= self.img_H_for_pred - 1)  # Bx1xN
        z_inside_mask = pc_px_py_pz[:, 2:3, :] > 0.1  # Bx1xN

        pc_coview_labels = (x_inside_mask & y_inside_mask & z_inside_mask).squeeze(1)  # BxN
        
        #----------cal class for each point
        KP_pc_pxpy_scale_int = torch.floor(pc_uv / 8.0).to(dtype=torch.long)
        KP_pc_pxpy_index = KP_pc_pxpy_scale_int[:, 0, :] + KP_pc_pxpy_scale_int[:, 1, :] * int(round(self.img_W / self.img_fine_resolution_scale))  # BxN
        #organize everything into (B*N)x* shape
        inside_mask_Bn = pc_coview_labels.reshape(self.B * self.N)  # BN
        inside_mask_Bn_int = inside_mask_Bn.to(dtype=torch.int32)  # BN
        insider_num = int(torch.sum(inside_mask_Bn_int).item())  # scalar int
        _, inside_idx_Bn = torch.sort(inside_mask_Bn_int, descending=True)  # BN
        insider_idx = inside_idx_Bn[0: insider_num]  # B_insider (B,)
        KP_pc_pxpy_index_Bn = KP_pc_pxpy_index.view(self.B*self.N)  # BN in long
        
        pc_cla_labels = torch.gather(KP_pc_pxpy_index_Bn, dim=0, index=insider_idx)  # B_insider in long 

        #----------cal pixel corrdinate for each point
        KP_pc_pxpy_int = torch.round(pc_uv)   #(B,2,N)
        KP_pc_pxpy_index_pixel = KP_pc_pxpy_int[:, 0, :] + KP_pc_pxpy_int[:, 1, :] * int(round(self.img_W_for_pred))
        KP_pc_pxpy_index_pixel_Bn = KP_pc_pxpy_index_pixel.view(self.B*self.N)
        #pixel-point match labels
        pc_pix_labels = torch.gather(KP_pc_pxpy_index_pixel_Bn, dim=0, index=insider_idx) # (B*insider_num,)

        # assure correctness
        fine_labels_min = torch.min(pc_cla_labels).item()
        fine_labels_max = torch.max(pc_cla_labels).item()
        assert fine_labels_min >= 0
        assert fine_labels_max <= self.img_W_fine_res * self.img_H_fine_res - 1

        return pc_coview_labels, pc_cla_labels, pc_pix_labels, insider_num, insider_idx

    def cal_acc(self,pc_coview_labels, pc_coview_pre, pc_cla_labels, fine_predictions_insider, insider_num):

        pc_coview_accuracy = torch.sum(torch.eq(pc_coview_labels.to(dtype=torch.long), pc_coview_pre).to(dtype=torch.float)) / (self.B * self.N)
        pc_class_accuracy = torch.sum(torch.eq(pc_cla_labels, fine_predictions_insider).to(dtype=torch.float)) / insider_num

        return pc_coview_accuracy.data.cpu().numpy(), pc_class_accuracy.data.cpu().numpy()

    def cal_rte_rre(self,
                    pc_inside_pred, 
                    pc_feat_in,
                    pc_insider_cla_pred,
                    img_pixel_inside_pred,
                    pixel_feat_in,
                    pixel_in):

        #------------------selecting features in co-view region------------------------------------------
        # pc_scores_Bn = pc_scores_flatten.reshape(self.B * self.N)
        # pc_inside_flatten = torch.where(pc_scores_Bn > 0.95)[0]
        # pc_inside_pred_num = pc_inside_flatten.size()[0]
        # pc_inside_flatten_BL = pc_inside_flatten.unsqueeze(1).expand(pc_inside_pred_num, self.L)
        # pc_cla_score = torch.gather(fine_scores_BnL, dim=0, index=pc_inside_flatten_BL)
        # _, pc_insider_cla_pred = torch.max(pc_cla_score, dim=1, keepdim=False)
        
        # #pc & pc_features in co-view
        # pc_inside_pred = self.pc[:, :, pc_inside_flatten]
        # pc_feat_in = self.pc_siam_feature_norm[:, :, pc_inside_flatten]

        #get predicted pixel inside and corresponding class
        # img_x_cla_label=torch.linspace(0,self.img_W_for_pred-1,self.img_W_for_pred).reshape(1,-1).repeat(self.img_H_for_pred,1).reshape(1,self.img_H_for_pred,self.img_W_for_pred).cuda()
        # img_y_cla_label=torch.linspace(0,self.img_H_for_pred-1,self.img_H_for_pred).reshape(-1,1).repeat(1, self.img_W_for_pred).reshape(1,self.img_H_for_pred,self.img_W_for_pred).cuda()
        # img_xy_flatten = (img_y_cla_label * self.img_W_for_pred  + img_x_cla_label).reshape(self.B*self.img_W_for_pred*self.img_H_for_pred)

        # img_xy_cla_label = torch.floor(img_y_cla_label / 8) * self.img_W_for_pred / 8  + torch.floor(img_x_cla_label / 8)
        # img_xy_cla_label=img_xy_cla_label.permute(1,2,0).squeeze(-1).unsqueeze(0).expand(self.B, self.img_H_for_pred, self.img_W_for_pred)
        # img_xy_cla_label_flatten = img_xy_cla_label.reshape(self.B *self.img_W_for_pred * self.img_H_for_pred)
        # img_score_flatten = self.img_score.reshape(self.B * self.img_W_for_pred * self.img_H_for_pred)

        # img_inside_index = torch.where(img_score_flatten > 0.95)[0]

        # img_pixel_inside_pred = torch.gather(img_xy_cla_label_flatten, dim=0, index=img_inside_index)
        # pixel_in = img_xy_flatten[img_inside_index]
        # pixel_feat_in = self.img_siam_feature_norm[:, :, img_inside_index].squeeze(0)
    
        #遍历一遍像素和点云,以每个class为键建立字典,找到预测的该class中inside的点云和像素
        match_dict_pc = {str(key) :[] for key in range(self.img_W_fine_res* self.img_H_fine_res) }
        match_dict_pixel = {str(key) :[] for key in range(self.img_W_fine_res* self.img_H_fine_res) }

        start_time = time.time()
        range_num = max(pc_insider_cla_pred.shape[0], img_pixel_inside_pred.shape[0])
        for index in range(range_num):
            if index < pc_insider_cla_pred.shape[0]:
                match_dict_pc[str(int(pc_insider_cla_pred[index]))].append(index)
            else:
                pass

            if index < img_pixel_inside_pred.shape[0]:
                match_dict_pixel[str(int(img_pixel_inside_pred[index]))].append(index)                                                  
            else:
                pass
        end_time = time.time()
        # print(end_time - start_time)
        #为每个class中的共视像素找到最匹配的点云
        pixel_match_index_list = []
        pc_match_index_list = []
        for key in match_dict_pc.keys():
            #get pc feature from index
            if len(match_dict_pc[key]) ==0 or len(match_dict_pixel[key]) == 0:
                # print('%s no point or pixel'%key)
                continue
            pc_siam_feature_key = pc_feat_in[:, :, match_dict_pc[key]].squeeze(0)
            #get pixel feature from index
            img_siam_feature_key = pixel_feat_in[:, :, match_dict_pixel[key]].squeeze(0)

            #cal correlation and select the most match one pc for each pixel

            #cosine distance
            correlation_map = 1 - torch.mm(img_siam_feature_key.t(), pc_siam_feature_key) 
            pixel_match_pc_index = torch.argmin(correlation_map, dim=1)
            #find match pixel and pc index
            pc_list = [match_dict_pc[key][index] for index in pixel_match_pc_index]
            pc_match_index_list += pc_list
            pixel_match_index_list += match_dict_pixel[key]

            #find match pixel and pc index
            pc_list = [match_dict_pc[key][index] for index in pixel_match_pc_index]
            pc_match_index_list += pc_list
            pixel_match_index_list += match_dict_pixel[key]
        


        pixel_matched = pixel_in[pixel_match_index_list]
        pixel_matched_y = torch.floor(pixel_matched / self.img_W_for_pred)
        pixel_matched_x = pixel_matched - pixel_matched_y * self.img_W_for_pred
        pixel_matched_xy = torch.cat([pixel_matched_x.unsqueeze(0), pixel_matched_y.unsqueeze(0)], dim=0)
        
        pc_matched_np = pc_inside_pred[:, :, pc_match_index_list].squeeze(0).data.cpu().numpy()
        pc_matched_np = pc_inside_pred[:, :, pc_match_index_list].squeeze(0).data.cpu().numpy()
        pixel_matched_xy_np = pixel_matched_xy.data.cpu().numpy()
        K_np = self.K.squeeze(0).data.cpu().numpy()
        P_np = self.P.squeeze(0).data.cpu().numpy()
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

        return t_diff, angles_diff

    def forward(self):

        pc_coview_labels, pc_cla_labels, pc_pix_labels,\
        insider_num, insider_idx = self.cal_labels_()

        #--------------------------------organize data------------------------------------------------
        pc_scores_flatten = self.pc_score.permute(0, 2, 1).squeeze(-1).contiguous()  # BNxL
        pc_coview_pre = torch.where(pc_scores_flatten > 0.95, 1.0, 0.0)
        # BxLxN -> BxNxL
        L = self.pc_class_scores.size(1)
        fine_scores_BnL = self.pc_class_scores.permute(0, 2, 1).reshape(self.B*self.N, L).contiguous()  # BNxL
        insider_idx_BinsiderL = insider_idx.unsqueeze(1).expand(insider_num, L)  # B_insiderxL
        fine_scores_insider = torch.gather(fine_scores_BnL, dim=0, index=insider_idx_BinsiderL)  # B_insiderxL
        _, fine_predictions_insider = torch.max(fine_scores_insider, dim=1, keepdim=False)

        #-------------------------------------------------------------------------------
        pc_scores_Bn = pc_scores_flatten.reshape(self.B*self.N)
        pc_inside_flatten = torch.where(pc_scores_Bn > 0.95)[0]
        pc_inside_pred_num = pc_inside_flatten.size()[0]
        pc_inside_flatten_BL = pc_inside_flatten.unsqueeze(1).expand(pc_inside_pred_num, L)
        pc_cla_score = torch.gather(fine_scores_BnL, dim=0, index=pc_inside_flatten_BL)
        pc_inside_pred = self.pc[:, :, pc_inside_flatten]
        pc_feat_in = self.pc_siam_feature_norm[:, :, pc_inside_flatten]
        _, pc_insider_cla_pred = torch.max(pc_cla_score, dim=1, keepdim=False)

        #get predicted pixel inside and corresponding class features
        img_x_cla_label=torch.linspace(0,self.img_W_for_pred-1,self.img_W_for_pred).reshape(1,-1).repeat(self.img_H_for_pred,1).reshape(1,self.img_H_for_pred,self.img_W_for_pred).cuda()
        img_y_cla_label=torch.linspace(0,self.img_H_for_pred-1,self.img_H_for_pred).reshape(-1,1).repeat(1, self.img_W_for_pred).reshape(1,self.img_H_for_pred,self.img_W_for_pred).cuda()
        img_xy_flatten = (img_y_cla_label * self.img_W_for_pred  + img_x_cla_label).reshape(self.B*self.img_W_for_pred*self.img_H_for_pred)

        img_xy_cla_label = torch.floor(img_y_cla_label / 8) * self.img_W_for_pred / 8  + torch.floor(img_x_cla_label / 8)
        img_xy_cla_label=img_xy_cla_label.permute(1,2,0).squeeze(-1).unsqueeze(0).expand(self.B, self.img_H_for_pred, self.img_W_for_pred)
        img_xy_cla_label_flatten = img_xy_cla_label.reshape(self.B *self.img_W_for_pred * self.img_H_for_pred)

        img_score_flatten = self.img_score.reshape(self.B * self.img_W_for_pred * self.img_H_for_pred)

        img_inside_index = torch.where(img_score_flatten > 0.95)[0]

        img_pixel_inside_pred = torch.gather(img_xy_cla_label_flatten, dim=0, index=img_inside_index)
        pixel_in = img_xy_flatten[img_inside_index]
        pixel_feat_in = self.img_siam_feature_norm[:, :, img_inside_index]

        #-------------------cal acc--------------------------------------
        pc_coview_accuracy, pc_class_accuracy  = self.cal_acc(pc_coview_labels, pc_coview_pre, pc_cla_labels, fine_predictions_insider, insider_num)

        #--------------------cal rre rte
        t_diff, angles_diff = self.cal_rte_rre(pc_inside_pred,
                                                pc_feat_in,
                                                pc_insider_cla_pred,
                                                img_pixel_inside_pred,
                                                pixel_feat_in,
                                                pixel_in)
        return pc_coview_accuracy, pc_class_accuracy, t_diff, angles_diff

