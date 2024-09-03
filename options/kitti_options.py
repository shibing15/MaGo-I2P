import numpy as np
import math
import torch


class KITTI_Options:
    def __init__(self):
        self.save_path = '/home/sunyunda/code_i2p/mambai2p/runs/mamba_cir_cla'

        #dataset-----------------------------------------
        self.dataset_name = 'KITTI'
        self.data_path = '/home/data/syd/KITTI_for_DEEPI2P'
        # CAM coordinate
        self.P_tx_amplitude = 10
        self.P_ty_amplitude = 0
        self.P_tz_amplitude = 10
        self.P_Rx_amplitude = 2 * math.pi * 0
        self.P_Ry_amplitude = 2 * math.pi
        self.P_Rz_amplitude = 2 * math.pi * 0
        self.dataloader_threads = 10
        self.crop_original_top_rows = 50
        self.img_scale = 0.5
        self.img_H = 160  # 320 * 0.5
        self.img_W = 512  # 1224 * 0.5
        # the fine resolution is img_H/scale x img_W/scale
        self.img_fine_resolution_scale = 32        
        self.input_pt_num = 40960
        self.pc_min_range = -1.0
        self.pc_max_range = 80.0
        self.node_a_num = 256
        self.node_b_num = 256
        # self.node_a_num = 128
        # self.node_b_num = 128
        self.k_ab = 32
        # self.k_ab = 64
        self.k_interp_ab = 3
        self.k_interp_point_a = 3
        self.k_interp_point_b = 3
        self.is_fine_resolution = True
        self.is_remove_ground = False

        #model----------------------------------------------
        self.mode_name = 'PointMamba'
        self.trans_dim = 384
        self.depth = 12
        self.cls_dim = 40
        self.num_heads = 6
        self.group_size = 32
        self.num_group = 256
        self.encoder_dims = 384
        self.rms_norm = False
        self.drop_path = 0.3
        self.drop_out = 0.
        self.drop_path_rate = 0.1
        self.fetch_idx = [3, 7, 11]
        #train-------------------------------------------------
        self.epoch = 25
        self.train_batch_size = 8
        self.val_batch_size = 4
        self.num_workers = 8
        self.is_debug = False
        self.num_kpt = 512
        self.dist_thres = 1
        #co_view threshold
        self.img_thres = 0.95
        self.pc_thres = 0.95
        #circle loss margin
        self.pos_margin=0.2
        self.neg_margin=1.8
        self.gpu_ids = [0]
        self.device = torch.device('cuda', self.gpu_ids[0])
        self.normalization = 'batch'
        self.norm_momentum = 0.1
        self.activation = 'relu'
        self.lr = 0.0001
        self.min_lr = 0.00001
        self.lr_decay_step = 20
        self.lr_decay_scale = 0.5
        self.vis_max_batch = 4




