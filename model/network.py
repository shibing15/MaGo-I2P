import torch
import torch.nn as nn
import torch.nn.functional as F
from . import imagenet
from .imagenet import ResidualConv,ImageUpSample
from .spvcnn.spvcnn import SPVCNN
from .pointmamba import PointMamba_Pointwise
from .kitti_options import KITTI_Options

# import imagenet
# from imagenet import ResidualConv,ImageUpSample
# from spvcnn.spvcnn import SPVCNN
# from pointmamba import PointMamba_Pointwise
# from kitti_options import KITTI_Options


class MambaI2P(nn.Module):
    def __init__(self, opt):
        super(MambaI2P, self).__init__()
        
        self.opt = opt
        self.H_fine_res = int(round(self.opt.img_H / self.opt.img_fine_resolution_scale))
        self.W_fine_res = int(round(self.opt.img_W / self.opt.img_fine_resolution_scale))

        #点云处理网络
        # self.pc_encoder = pointnet2.PCEncoder(opt, Ca=64, Cb=256, Cg=512)
        # self.voxel_branch = SPVCNN(num_classes=128, cr=0.5, pres=0.05, vres=0.05)
        self.mamba = PointMamba_Pointwise(opt)
       
        #图像处理网络
        self.img_encoder = imagenet.ImageEncoder()
       
        #submodel
        self.pc_score_head=nn.Sequential(
            nn.Conv1d(128+512,128,1,bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,64,1,bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,1,1,bias=False),
            nn.Sigmoid())

        self.img_score_head=nn.Sequential(
            nn.Conv2d(64+512,128,1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,64,1,bias=False),
            nn.BatchNorm2d(64),nn.ReLU(),
            nn.Conv2d(64,1,1,bias=False),
            nn.Sigmoid())

        self.img_feature_layer=nn.Sequential(nn.Conv2d(64,64,1,bias=False),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,64,1,bias=False),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,64,1,bias=False))
        self.pc_feature_layer=nn.Sequential(nn.Conv1d(128,128,1,bias=False),nn.BatchNorm1d(128),nn.ReLU(),nn.Conv1d(128,128,1,bias=False),nn.BatchNorm1d(128),nn.ReLU(),nn.Conv1d(128,64,1,bias=False))
        


    def forward(self, pc, intensity, sn, img):

        
        #para inital
        B,N,=pc.size(0),pc.size(2)
        #--------------------------------------feture extration-----------------------------------
        #---------------------point_mammba--------------------------
        global_pc_feat, point_wise_feat = self.mamba(pc)
        #-----------------------voxel_pc----------------------------
        # input_voxel = torch.cat((pc, intensity, sn), dim=1).transpose(2,1).reshape(-1, 7)
        # batch_inds = torch.arange(pc.shape[0]).reshape(-1,1).repeat(1,pc.shape[2]).reshape(-1, 1).cuda()
        # corrds = pc.transpose(2,1) - torch.min(pc.transpose(2,1), dim=1, keepdim=True)[0]
        # corrds = corrds.reshape(-1, 3)
        # corrds = torch.round(corrds / 0.05)
        # corrds = torch.cat((corrds, batch_inds), dim=-1)
        # _, voxel_feat = self.voxel_branch(input_voxel, corrds, pc.shape[0])
        #-----------------------img_brach--------------------------
        global_img_feat, pixel_wise_feat=self.img_encoder(img)
        # point_wise_feat = point_wise_feat + voxel_feat

        #------------------------------fuse------------------------

        img_feat_fusion = torch.cat((pixel_wise_feat, global_pc_feat.unsqueeze(-1).unsqueeze(-1).repeat(1,1,pixel_wise_feat.shape[2],pixel_wise_feat.shape[3])), dim=1)
        pc_feat_fusion = torch.cat((point_wise_feat, global_img_feat.unsqueeze(-1).repeat(1,1,point_wise_feat.shape[2])), dim=1)
        #----------------------------------------------------------------------
        img_score = self.img_score_head(img_feat_fusion)    #[2, 1, 40, 128]
        pc_score = self.pc_score_head(pc_feat_fusion)   #[2, 1, 40960]

        pixel_wise_feat = self.img_feature_layer(pixel_wise_feat) 
        point_wise_feat = self.pc_feature_layer(point_wise_feat)  
        point_wise_feat=F.normalize(point_wise_feat, dim=1,p=2)   #[2, 64, 40, 128]
        pixel_wise_feat=F.normalize(pixel_wise_feat, dim=1,p=2)   #[2, 64, 40960]
        pixel_wise_feat_flatten = pixel_wise_feat.flatten(start_dim=2)
        img_score_flatten = img_score.flatten(start_dim=2)

        return pixel_wise_feat_flatten, point_wise_feat, img_score_flatten, pc_score

class Point_classifier(nn.Module):
    def __init__(self, input_channel, img_W, img_H):
        super().__init__()
        self.convs1 = nn.Conv1d(input_channel, 512, 1)
        self.dp1 = nn.Dropout(0.5)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, img_W*img_H, 1)
        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
    
    def forward(self, pc_features):
        x = self.relu(self.bns1(self.convs1(pc_features)))
        x = self.dp1(x)
        x = self.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)
        x = F.log_softmax(x, dim=1)
        
        return x

class MambaI2P_cla(nn.Module):
    def __init__(self, opt):
        super(MambaI2P_cla, self).__init__()
        
        self.opt = opt
        self.H_fine_res = int(round(self.opt.img_H / self.opt.img_fine_resolution_scale))
        self.W_fine_res = int(round(self.opt.img_W / self.opt.img_fine_resolution_scale))

        #点云处理网络
        # self.pc_encoder = pointnet2.PCEncoder(opt, Ca=64, Cb=256, Cg=512)
        # self.voxel_branch = SPVCNN(num_classes=128, cr=0.5, pres=0.05, vres=0.05)
        self.mamba = PointMamba_Pointwise(opt)
       
        #图像处理网络
        self.img_encoder = imagenet.ImageEncoder()
       
        #submodel
        self.pc_score_head=nn.Sequential(
            nn.Conv1d(128+512,128,1,bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,64,1,bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,1,1,bias=False),
            nn.Sigmoid())

        self.img_score_head=nn.Sequential(
            nn.Conv2d(64+512,128,1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,64,1,bias=False),
            nn.BatchNorm2d(64),nn.ReLU(),
            nn.Conv2d(64,1,1,bias=False),
            nn.Sigmoid())

        self.pc_classifier = Point_classifier(128+512, self.W_fine_res, self.H_fine_res)

        self.img_feature_layer=nn.Sequential(
            nn.Conv2d(64+512,128,1,bias=False),
            nn.BatchNorm2d(128),nn.ReLU(),
            nn.Conv2d(128,64,1,bias=False),
            nn.BatchNorm2d(64),nn.ReLU(),
            nn.Conv2d(64,64,1,bias=False))

        self.pc_feature_layer=nn.Sequential(
            nn.Conv1d(128+512,256,1,bias=False),
            nn.BatchNorm1d(256),nn.ReLU(),
            nn.Conv1d(256,128,1,bias=False),
            nn.BatchNorm1d(128),nn.ReLU(),
            nn.Conv1d(128,64,1,bias=False))
        


    def forward(self, pc, intensity, sn, img):

        
        #para inital
        B,N,=pc.size(0),pc.size(2)
        #--------------------------------------feture extration-----------------------------------
        #---------------------point_mammba--------------------------
        global_pc_feat, point_wise_feat = self.mamba(pc)
        #-----------------------voxel_pc----------------------------
        # input_voxel = torch.cat((pc, intensity, sn), dim=1).transpose(2,1).reshape(-1, 7)
        # batch_inds = torch.arange(pc.shape[0]).reshape(-1,1).repeat(1,pc.shape[2]).reshape(-1, 1).cuda()
        # corrds = pc.transpose(2,1) - torch.min(pc.transpose(2,1), dim=1, keepdim=True)[0]
        # corrds = corrds.reshape(-1, 3)
        # corrds = torch.round(corrds / 0.05)
        # corrds = torch.cat((corrds, batch_inds), dim=-1)
        # _, voxel_feat = self.voxel_branch(input_voxel, corrds, pc.shape[0])
        #-----------------------img_brach--------------------------
        global_img_feat, pixel_wise_feat=self.img_encoder(img)
        # point_wise_feat = point_wise_feat + voxel_feat

        #------------------------------fuse------------------------

        img_feat_fusion = torch.cat((pixel_wise_feat, global_pc_feat.unsqueeze(-1).unsqueeze(-1).repeat(1,1,pixel_wise_feat.shape[2],pixel_wise_feat.shape[3])), dim=1)
        pc_feat_fusion = torch.cat((point_wise_feat, global_img_feat.unsqueeze(-1).repeat(1,1,point_wise_feat.shape[2])), dim=1)
        #----------------------------------------------------------------------
        img_score = self.img_score_head(img_feat_fusion)    #[2, 1, 40, 128]
        pc_score = self.pc_score_head(pc_feat_fusion)   #[2, 1, 40960]

        pc_cla_scores = self.pc_classifier(pc_feat_fusion)

        pixel_wise_feat = self.img_feature_layer(img_feat_fusion) 
        point_wise_feat = self.pc_feature_layer(pc_feat_fusion)  
        point_wise_feat=F.normalize(point_wise_feat, dim=1,p=2)   #[2, 64, 40, 128]
        pixel_wise_feat=F.normalize(pixel_wise_feat, dim=1,p=2)   #[2, 64, 40960]
        pixel_wise_feat_flatten = pixel_wise_feat.flatten(start_dim=2)
        img_score_flatten = img_score.flatten(start_dim=2)


        return pixel_wise_feat_flatten, point_wise_feat, img_score_flatten, pc_score, pc_cla_scores


if __name__=='__main__':
    opt=KITTI_Options()
    pc=torch.rand(10,3,20480).cuda()
    node_a = torch.rand(10,3,256).cuda()
    node_b = torch.rand(10,3,256).cuda()
    intensity=torch.rand(10,1,20480).cuda()
    sn=torch.rand(10,3,20480).cuda()
    img=torch.rand(10,4,160,512).cuda()
    net=MambaI2P_cla(opt).cuda()
    pixel_wise_feat_flatten, point_wise_feat, img_score_flatten, pc_score, pc_cla_scores=net(pc,intensity,sn,img)
    import ipdb;ipdb.set_trace()
    print(1)

    