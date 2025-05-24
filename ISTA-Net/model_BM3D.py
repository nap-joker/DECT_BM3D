import torch
import torch.nn as nn
import numpy as np
from scipy.fftpack import dct, idct
import cv2

class BM3D(nn.Module):
    def __init__(self, lamb2d=2.0, lamb3d=2.7):
        super(BM3D, self).__init__()
        self.lamb2d = lamb2d
        self.lamb3d = lamb3d
        
        # Step 1 parameters
        self.step1_thre_dist = 3000
        self.step1_max_match = 32
        self.step1_block_size = 16
        self.step1_spdup_factor = 3
        self.step1_window_size = 39
        
        # Step 2 parameters
        self.step2_thre_dist = 500
        self.step2_max_match = 64
        self.step2_block_size = 16
        self.step2_spdup_factor = 4
        self.step2_window_size = 39
        
        self.kaiser_window_beta = 2.0
        
    def estimate_noise(self,img):
        denoised = cv2.medianBlur(img,3)
        diff = img.astype(np.float32)-denoised.astype(np.float32)
        mad = np.median(np.abs(diff - np.median(diff)))
        sigma = mad / 0.6745
        return sigma
        
    def forward(self, x):
        # Convert PyTorch tensor to numpy array
        x_np = x.detach().cpu().numpy()
        x_np = np.squeeze(x_np)  # Remove batch and channel dimensions
        
        # Apply BM3D
        basic_img = self.bm3d_step1(x_np)
        final_img = self.bm3d_step2(basic_img, x_np)
        
        # Convert back to PyTorch tensor
        final_img = torch.from_numpy(final_img).float()
        final_img = final_img.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        final_img = final_img.to(x.device)
        
        return final_img

    def bm3d_step1(self, noisy_img):
        # Initialize
        block_size = self.step1_block_size
        basic_img, basic_weight, basic_kaiser = self.initialization(noisy_img, block_size, self.kaiser_window_beta)
        block_dct_all = self.pre_dct(noisy_img, block_size)

        # Block-wise estimate
        for i in range(int((noisy_img.shape[0]-block_size)/self.step1_spdup_factor)+2):
            for j in range(int((noisy_img.shape[1]-block_size)/self.step1_spdup_factor)+2):
                ref_point = [min(self.step1_spdup_factor*i, noisy_img.shape[0]-block_size-1),
                           min(self.step1_spdup_factor*j, noisy_img.shape[1]-block_size-1)]
                
                block_pos, block_group = self.step1_grouping(noisy_img, ref_point, block_dct_all, block_size,
                                                           self.step1_thre_dist, self.step1_max_match,
                                                           self.step1_window_size)
                
                block_group, nonzero_cnt = self.step1_3dfiltering(block_group)
                self.step1_aggregation(block_group, block_pos, basic_img, basic_weight, basic_kaiser, nonzero_cnt)

        basic_weight = np.where(basic_weight == 0, 1, basic_weight)
        basic_img[:, :] /= basic_weight[:, :]
        
        return basic_img

    def bm3d_step2(self, basic_img, noisy_img):
        # Initialize
        block_size = self.step2_block_size
        final_img, final_weight, final_kaiser = self.initialization(basic_img, block_size, self.kaiser_window_beta)
        block_dct_noisy = self.pre_dct(noisy_img, block_size)
        block_dct_basic = self.pre_dct(basic_img, block_size)

        # Block-wise estimate
        for i in range(int((basic_img.shape[0]-block_size)/self.step2_spdup_factor)+2):
            for j in range(int((basic_img.shape[1]-block_size)/self.step2_spdup_factor)+2):
                ref_point = [min(self.step2_spdup_factor*i, basic_img.shape[0]-block_size-1),
                           min(self.step2_spdup_factor*j, basic_img.shape[1]-block_size-1)]
                
                block_pos, block_group_basic, block_group_noisy = self.step2_grouping(
                    basic_img, noisy_img, ref_point, block_size, self.step2_thre_dist,
                    self.step2_max_match, self.step2_window_size, block_dct_basic, block_dct_noisy)
                
                block_group_noisy, wiener_weight = self.step2_3dfiltering(block_group_basic, block_group_noisy)
                self.step2_aggregation(block_group_noisy, wiener_weight, block_pos, final_img,
                                     final_weight, final_kaiser)

        final_weight = np.where(final_weight == 0, 1, final_weight)
        final_img[:, :] /= final_weight[:, :]
        
        return final_img

    def initialization(self, img, block_size, kaiser_window_beta):
        init_img = np.zeros(img.shape, dtype=float)
        init_weight = np.zeros(img.shape, dtype=float)
        window = np.matrix(np.kaiser(block_size, kaiser_window_beta))
        init_kaiser = np.array(window.T * window)
        return init_img, init_weight, init_kaiser

    def pre_dct(self, img, block_size):
        block_dct_all = np.zeros((img.shape[0]-block_size, img.shape[1]-block_size, block_size, block_size),
                                dtype=float)
        for i in range(block_dct_all.shape[0]):
            for j in range(block_dct_all.shape[1]):
                block = img[i:i+block_size, j:j+block_size]
                block_dct_all[i, j, :, :] = dct(dct(block.astype(np.float64), axis=0, norm='ortho'),
                                              axis=1, norm='ortho')
        return block_dct_all

    def step1_grouping(self, noisy_img, ref_point, block_dct_all, block_size, thre_dist, max_match, window_size):
        window_loc = self.search_window(noisy_img, ref_point, block_size, window_size)
        block_num_searched = (window_size-block_size+1)**2
        block_pos = np.zeros((block_num_searched, 2), dtype=int)
        block_group = np.zeros((block_num_searched, block_size, block_size), dtype=float)
        dist = np.zeros(block_num_searched, dtype=float)
        ref_dct = block_dct_all[ref_point[0], ref_point[1], :, :]
        match_cnt = 0

        for i in range(window_size-block_size+1):
            for j in range(window_size-block_size+1):
                searched_dct = block_dct_all[window_loc[0, 0]+i, window_loc[0, 1]+j, :, :]
                dist_val = self.step1_compute_dist(ref_dct, searched_dct)
                if dist_val < thre_dist:
                    block_pos[match_cnt, :] = [window_loc[0, 0]+i, window_loc[0, 1]+j]
                    block_group[match_cnt, :, :] = searched_dct
                    dist[match_cnt] = dist_val
                    match_cnt += 1

        if match_cnt <= max_match:
            block_pos = block_pos[:match_cnt, :]
            block_group = block_group[:match_cnt, :, :]
        else:
            idx = np.argpartition(dist[:match_cnt], max_match)
            block_pos = block_pos[idx[:max_match], :]
            block_group = block_group[idx[:max_match], :]

        return block_pos, block_group

    def step1_compute_dist(self, block_dct1, block_dct2):
        if block_dct1.shape != block_dct2.shape:
            raise ValueError("DCT blocks must have the same shape")
        block_size = block_dct1.shape[0]
        if self.sigma > 40:
            thre_value = self.lamb2d * self.sigma
            block_dct1 = np.where(np.abs(block_dct1) < thre_value, 0, block_dct1)
            block_dct2 = np.where(np.abs(block_dct2) < thre_value, 0, block_dct2)
        return np.linalg.norm(block_dct1 - block_dct2)**2 / (block_size**2)

    def step1_3dfiltering(self, block_group):
        thre_value = self.lamb3d * self.sigma
        nonzero_cnt = 0
        for i in range(block_group.shape[1]):
            for j in range(block_group.shape[2]):
                third_vector = dct(block_group[:, i, j], norm='ortho')
                third_vector[np.abs(third_vector[:]) < thre_value] = 0.
                nonzero_cnt += np.nonzero(third_vector)[0].size
                block_group[:, i, j] = list(idct(third_vector, norm='ortho'))
        return block_group, nonzero_cnt

    def step1_aggregation(self, block_group, block_pos, basic_img, basic_weight, basic_kaiser, nonzero_cnt):
        if nonzero_cnt < 1:
            block_weight = 1.0 * basic_kaiser
        else:
            block_weight = (1./(self.sigma**2 * nonzero_cnt)) * basic_kaiser

        for i in range(block_pos.shape[0]):
            basic_img[block_pos[i, 0]:block_pos[i, 0]+block_group.shape[1],
                     block_pos[i, 1]:block_pos[i, 1]+block_group.shape[2]] += \
                block_weight * idct(idct(block_group[i, :, :], axis=0, norm='ortho'), axis=1, norm='ortho')
            basic_weight[block_pos[i, 0]:block_pos[i, 0]+block_group.shape[1],
                        block_pos[i, 1]:block_pos[i, 1]+block_group.shape[2]] += block_weight

    def step2_grouping(self, basic_img, noisy_img, ref_point, block_size, thre_dist, max_match, window_size,
                      block_dct_basic, block_dct_noisy):
        window_loc = self.search_window(basic_img, ref_point, block_size, window_size)
        block_num_searched = (window_size-block_size+1)**2
        block_pos = np.zeros((block_num_searched, 2), dtype=int)
        block_group_basic = np.zeros((block_num_searched, block_size, block_size), dtype=float)
        block_group_noisy = np.zeros((block_num_searched, block_size, block_size), dtype=float)
        dist = np.zeros(block_num_searched, dtype=float)
        match_cnt = 0

        for i in range(window_size-block_size+1):
            for j in range(window_size-block_size+1):
                searched_point = [window_loc[0, 0]+i, window_loc[0, 1]+j]
                dist_val = self.step2_compute_dist(basic_img, ref_point, searched_point, block_size)
                if dist_val < thre_dist:
                    block_pos[match_cnt, :] = searched_point
                    dist[match_cnt] = dist_val
                    match_cnt += 1

        if match_cnt <= max_match:
            block_pos = block_pos[:match_cnt, :]
        else:
            idx = np.argpartition(dist[:match_cnt], max_match)
            block_pos = block_pos[idx[:max_match], :]

        for i in range(block_pos.shape[0]):
            similar_point = block_pos[i, :]
            block_group_basic[i, :, :] = block_dct_basic[similar_point[0], similar_point[1], :, :]
            block_group_noisy[i, :, :] = block_dct_noisy[similar_point[0], similar_point[1], :, :]

        block_group_basic = block_group_basic[:block_pos.shape[0], :, :]
        block_group_noisy = block_group_noisy[:block_pos.shape[0], :, :]

        return block_pos, block_group_basic, block_group_noisy

    def step2_compute_dist(self, img, point1, point2, block_size):
        block1 = img[point1[0]:point1[0]+block_size, point1[1]:point1[1]+block_size].astype(np.float64)
        block2 = img[point2[0]:point2[0]+block_size, point2[1]:point2[1]+block_size].astype(np.float64)
        return np.linalg.norm(block1-block2)**2 / (block_size**2)

    def step2_3dfiltering(self, block_group_basic, block_group_noisy):
        weight = 0
        coef = 1.0 / block_group_noisy.shape[0]
        for i in range(block_group_noisy.shape[1]):
            for j in range(block_group_noisy.shape[2]):
                vec_basic = dct(block_group_basic[:, i, j], norm='ortho')
                vec_noisy = dct(block_group_noisy[:, i, j], norm='ortho')
                vec_value = vec_basic**2 * coef
                vec_value /= (vec_value + self.sigma**2)
                vec_noisy *= vec_value
                weight += np.sum(vec_value)
                block_group_noisy[:, i, j] = list(idct(vec_noisy, norm='ortho'))

        if weight > 0:
            wiener_weight = 1./(self.sigma**2 * weight)
        else:
            wiener_weight = 1.0

        return block_group_noisy, wiener_weight

    def step2_aggregation(self, block_group_noisy, wiener_weight, block_pos, final_img, final_weight, final_kaiser):
        block_weight = wiener_weight * final_kaiser
        for i in range(block_pos.shape[0]):
            final_img[block_pos[i, 0]:block_pos[i, 0]+block_group_noisy.shape[1],
                     block_pos[i, 1]:block_pos[i, 1]+block_group_noisy.shape[2]] += \
                block_weight * idct(idct(block_group_noisy[i, :, :], axis=0, norm='ortho'), axis=1, norm='ortho')
            final_weight[block_pos[i, 0]:block_pos[i, 0]+block_group_noisy.shape[1],
                        block_pos[i, 1]:block_pos[i, 1]+block_group_noisy.shape[2]] += block_weight

    def search_window(self, img, ref_point, block_size, window_size):
        if block_size >= window_size:
            raise ValueError("Block size must be smaller than window size")
        margin = np.zeros((2,2), dtype=int)
        margin[0, 0] = max(0, ref_point[0]+int((block_size-window_size)/2))
        margin[0, 1] = max(0, ref_point[1]+int((block_size-window_size)/2))
        margin[1, 0] = margin[0, 0] + window_size
        margin[1, 1] = margin[0, 1] + window_size

        if margin[1, 0] >= img.shape[0]:
            margin[1, 0] = img.shape[0] - 1
            margin[0, 0] = margin[1, 0] - window_size
        if margin[1, 1] >= img.shape[1]:
            margin[1, 1] = img.shape[1] - 1
            margin[0, 1] = margin[1, 1] - window_size

        return margin