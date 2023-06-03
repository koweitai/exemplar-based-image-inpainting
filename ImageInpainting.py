import cv2
import numpy as np
import argparse
from skimage.metrics import structural_similarity as ssim

alpha = 255 # normalization factor

# class Image:
#     def __init__(self, pixels, contour):
#         self.pixels = pixels
#         self.contour = contour

class Pixel:
    def __init__(self, r, c, value, is_filled):
        self.r = r
        self.c = c
        self.value = value # [B, G, R]
        self.confidence = 1 if is_filled else 0
        self.is_filled = is_filled # filled or not
        self.data = 0
        self.gradient = 0

    def set_patch(self, patch):
        self.patch = patch

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors # 九宮格的value

    def compute_confidence(self):
        """ Compute confidence of the central pixel of the patch. """
        confidence_sum = 0
        for row in self.patch:
            for ele in row:
                if ele.is_filled:
                    confidence_sum += ele.confidence
        confidence = confidence_sum / (self.patch.shape[0]*self.patch.shape[1]) # 面積應該是self.patch.shape[0]*self.patch.shape[1]?
        # print(f"[{self.r}, {self.c}]'s confidence: {self.confidence}")    
        return confidence
    
    def compute_data(self):
        """ Compute the data on linear structures around the pixel. """
        norm = self.normal_direction()
        gradient = self.gradient_vector()
        data = np.abs(norm.dot(gradient)) / alpha
        self.data = data
        return data

    def compute_patch_priority(self):
        return self.compute_confidence() * self.compute_data()
    
    def gradient_vector_old(self, k = 2): # per pixel (Sobel)
        max_gradient = -1
        for i in range(self.patch.shape[0]):
            for j in range(self.patch.shape[1]):
                point = self.patch[i][j]
                gr = []
                gc = []
                # G = [0 for _ in range(3)]
                for channel in range(3):
                    gr.append(((point.neighbors[0][2].value[channel] + k * point.neighbors[1][2].value[channel] + point.neighbors[2][2].value[channel]) - (point.neighbors[0][0].value[channel] + k * point.neighbors[1][0].value[channel] + point.neighbors[2][0].value[channel])) / (k + 2))
                    gc.append(((point.neighbors[0][0].value[channel] + k * point.neighbors[0][1].value[channel] + point.neighbors[0][2].value[channel]) - (point.neighbors[2][0].value[channel] + k * point.neighbors[2][1].value[channel] + point.neighbors[2][2].value[channel])) / (k + 2))
                    # G[channel] = np.hypot(gr, gc) # sqrt(x*x + y*y)
                # gradient = sum(G) / 3
                gr_sum = sum(gr) / 3
                gc_sum = sum(gc) / 3
                gradient = np.hypot(gr_sum, gc_sum) # sqrt(x*x + y*y)
                if gradient > max_gradient:
                    max_gradient = gradient
                    max_gradient_vec = np.array([gc_sum, gr_sum])
        magnitude = np.sqrt(max_gradient_vec.dot(max_gradient_vec))
        # gradient_vec = max_gradient_vec / magnitude * max_gradient # normalize?
        gradient_vec = max_gradient_vec
        self.gradient = magnitude
        return gradient_vec
    
    def gradient_vector(self, k = 2, printValue = False): # per pixel (Sobel)
        values = [np.empty(self.patch.shape) for _ in range(3)]
        
        for i in range(self.patch.shape[0]):
            for j in range(self.patch.shape[1]):
                for channel in range(3):
                    if self.patch[i][j].is_filled:
                        values[channel][i][j] = self.patch[i][j].value[channel]
                    else:
                        values[channel][i][j] = np.nan

        gr = []
        gc = []
        for channel in range(3):
            gr_here, gc_here = np.gradient(values[channel], axis=(0, 1))
            gr.append(gr_here)
            gc.append(gc_here)
        gr_sum = sum(gr) / 3
        gc_sum = sum(gc) / 3
        max_gradient = -1
        max_gradient_vec = np.array([0, 0])
        for i in range(gr_sum.shape[0]):
            for j in range(gr_sum.shape[1]):
                if not (np.isnan(gr_sum[i, j])) and not (np.isnan(gc_sum[i, j])):
                    gradient = np.sqrt(gr_sum[i, j]**2 + gc_sum[i, j]**2)
                else:
                    gradient = -1
                
                if gradient > max_gradient:
                    max_gradient = gradient
                    max_gradient_vec = np.array([gc_sum[i, j], gr_sum[i, j]])
        magnitude = np.sqrt(max_gradient_vec.dot(max_gradient_vec))
        # gradient_vec = max_gradient_vec / magnitude * max_gradient # normalize?
        gradient_vec = max_gradient_vec
        self.gradient = magnitude
        if printValue:
            print(f"this patch's gradient: {gradient_vec}")
        return gradient_vec
    
    def normal_direction(self, printValue = False):
        for point in contour_point:
            if point == [self.r, self.c]:
                cur_point = point
                break
        
        prev_point = [-1, -1]
        next_point = [-1, -1]
        for point in contour_point:
            if point == cur_point:
                continue
            dist = (point[0]-cur_point[0])**2 + (point[1]-cur_point[1])**2
            # if dist <= 2:
            if dist <= 3:
                if prev_point == [-1, -1]:
                    prev_point = point
                elif next_point == [-1, -1]:
                    next_point = point
                    break
        normal = np.array([prev_point[1]-next_point[1], prev_point[0]-next_point[0]])
        magnitude = np.sqrt(normal.dot(normal))
        normal = normal / magnitude * alpha  # normalize?
        if printValue:
            print(f"this patch's normal: {normal}")
        return normal

# Tool function
def is_contour(pixel):
    if pixel.is_filled == 1: # 已經填滿的區域
        return False
    if pixel.r-1 < 0 or pixel.r+1 >= shape[0] or pixel.c-1 < 0 or pixel.c+1 >= shape[1]: # 該要填滿點在圖的邊邊，視作邊緣
        return True
    if pixel.neighbors[0, 1].is_filled == 1 or pixel.neighbors[1, 2].is_filled == 1 or pixel.neighbors[2, 1].is_filled == 1 or pixel.neighbors[1, 0].is_filled == 1: # 上下左右有已經填滿的區域
        return True
    return False

# Method fuction
def init_mask(mask_img):
    output_image = np.ones_like(mask_img)
    for i in range(mask_img.shape[0]):
        for j in range(mask_img.shape[1]):
            output_image[i, j] = 0 if mask_img[i, j] < 128 else 255
    return output_image

def init_image(img_input, img_mask):
    img = np.empty(shape=img_input.shape[0:2], dtype = np.dtype(Pixel))
    global contour_point
    contour_point = []
    for i in range(img_mask.shape[0]):
        for j in range(img_mask.shape[1]): 
            pixel = Pixel(i, j, img_input[i][j] if img_mask[i][j] == 0 else [0, 0, 0], img_mask[i][j] == 0) 
            img[i][j] = pixel

    for i in range(img_mask.shape[0]):
        for j in range(img_mask.shape[1]):
            patch = img[max(i-patch_size//2, 0):min(i+1+patch_size//2, img_mask.shape[0]), max(j-patch_size//2, 0):min(j+1+patch_size//2, img_mask.shape[1])]
            img[i][j].set_patch(patch)
            neighbors = np.empty(shape=[3, 3], dtype = np.dtype(Pixel))
            for i_patch in range(3):
                for j_patch in range(3):
                    if i+i_patch-1 > 0 and i+i_patch-1 < img_input.shape[0] and j+j_patch-1 > 0 and j+j_patch-1 < img_input.shape[1]:
                        neighbors[i_patch][j_patch] = img[i+i_patch-1][j+j_patch-1]
                    else:
                        pixel = Pixel(-1, -1, [-1, -1, -1], -1) # Should not be -1? 算gradient會有問題
                        neighbors[i_patch][j_patch] = pixel
            img[i][j].set_neighbors(neighbors)

    update_contour_point(img)
    
    return img

def find_maxpriority_patch(img):
    max_priority = -100
    max_priority_point_idx = contour_point[0]
    for idx in contour_point: # idx = [i, j]
        point = img[idx[0], idx[1]] # point is a Pixel
        pri = point.compute_patch_priority()
        if pri > max_priority:
            # print(f'{idx}: {pri}') 
            max_priority = pri
            max_priority_point_idx = idx

    return max_priority_point_idx

def compute_difference(target_patch, source_patch):
    # target_patch 中會有很多是待填滿的點，在比較的時候是不是只比較已填或不用填的點們去跟 source_patch 比？
    # 確保 source_patch 裡面每個點都是有顏色的（填滿的 -> 只要target_patch沒有被填的點，source_patch都已經被填滿
    min_SSIM = -1 # if source_patch 不是填滿的

    if source_patch.shape != target_patch.shape:
        return min_SSIM
    
    img_target = np.zeros([patch_size, patch_size, 3], dtype=np.uint8)
    img_source = np.zeros([patch_size, patch_size, 3], dtype=np.uint8)
    for i in range(target_patch.shape[0]):
        for j in range(target_patch.shape[1]):
            img_source[i, j] = source_patch[i, j].value
            if target_patch[i, j].is_filled: # 只看 target_patch 有填的點
                img_target[i, j] = target_patch[i, j].value # [B, G, R]
            else: # target_patch沒有被填的點，source_patch必須已經被填滿
                if not source_patch[i, j].is_filled:
                    return min_SSIM
                img_target[i, j] = source_patch[i, j].value # 用source_patch的值代替
    ssim_value = ssim(img_target, img_source, multichannel=True, win_size=patch_size, channel_axis=2)
    # print(ssim_value)
    return ssim_value

def find_source_patch(target_patch_point_idx, img):
    target_patch = img[target_patch_point_idx[0], target_patch_point_idx[1]].patch
    max_SSIM = -1
    max_SSIM_patch = target_patch
    for row in img:
        for ele in row: # ele is a Pixel
            source_patch = ele.patch
            difference = compute_difference(target_patch, source_patch)
            if  difference > max_SSIM:
                max_SSIM_patch = source_patch
                max_SSIM = difference

    return max_SSIM_patch

def fill_imagedata(target_patch_pixel, source_patch):
    # update confidence here?
    target_patch_pixel.confidence = target_patch_pixel.compute_confidence()
    # print(target_patch_pixel.confidence, target_patch_pixel.gradient, target_patch_pixel.data)
    for i in range(source_patch.shape[0]):
        for j in range(source_patch.shape[1]):
            if not target_patch_pixel.patch[i][j].is_filled:
                target_patch_pixel.patch[i][j].value = source_patch[i][j].value
                target_patch_pixel.patch[i][j].is_filled = True
                target_patch_pixel.patch[i][j].confidence = target_patch_pixel.confidence
    return

def update_contour_point(img):
    # 為了算邊緣線上點的法向量，找所有邊緣曲線的點放進 contour_point，而且要 contour_point 裏的順序是連著線的
    # 要處理很多個區塊要補的狀況！！！
    # 要處理填滿隙縫的問題！
    contour_point.clear()
    for i in range(shape[0]):
        for j in range(shape[1]): 
            if not img[i][j].is_filled and is_contour(img[i][j]) and [i, j] not in contour_point:
                contour_point.append([i, j])
    return

def generate_result_image_test(img_input, img, point_idx, source_patch): # 單純測試有沒有找到欲填範圍的邊緣
    img_result = np.zeros(img_input.shape, dtype=np.uint8)
    # img_confidence = np.zeros(img_input.shape[:-1], dtype=np.uint8)
    # img_data = np.zeros(img_input.shape[:-1], dtype=np.uint8)
    # img_gradient = np.zeros(img_input.shape[:-1], dtype=np.uint8)
    # max_magnitude = -1
    # max_data = -1
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # img_confidence[i, j] = int(img[i, j].confidence * 255)
            # img_data[i, j] = np.clip(img[i, j].data, 0, 255)
            # img_gradient[i, j] = np.clip(img[i, j].gradient, 0, 255)
            if [i, j] in contour_point:
                img_result[i, j] = [255, 0, 0]
                # norm = img[i][j].normal_direction()
                # gradient = img[i][j].gradient_vector()
                # data = img[i][j].compute_data()
                # print(i, j, norm, gradient, data)
                # magnitude = np.sqrt(gradient.dot(gradient))
                # if magnitude > max_magnitude:
                #     max_magnitude = magnitude
                #     max_magnitude_vec = gradient
                # if data > max_data:
                #     max_data = data
                #     max_data_norm = norm
                #     max_data_gradient = gradient
            elif not img[i, j].is_filled:
                img_result[i, j] = [255, 255, 255]
            else:
                img_result[i, j] = img[i, j].value
    # for source_patch in source_patches:
    for i in range(source_patch.shape[0]):
        img_result[source_patch[i][0].r, source_patch[i][0].c] = [0, 255, 0]
        img_result[source_patch[i][source_patch.shape[1]-1].r, source_patch[i][source_patch.shape[1]-1].c] = [0, 255, 0]
    for j in range(source_patch.shape[1]):
        img_result[source_patch[0][j].r, source_patch[0][j].c] = [0, 255, 0]
        img_result[source_patch[source_patch.shape[0]-1][j].r, source_patch[source_patch.shape[0]-1][j].c] = [0, 255, 0]
    # print(max_magnitude, max_magnitude_vec)
    # print(max_data, max_data_norm, max_data_gradient)
    # for point_idx in point_idxs:
    img_result[point_idx[0], point_idx[1]] = [0, 0, 255]
    # max_data會到 265.30946553035, norm=[114.03946685 228.0789337 ], gradient=[111.75 240.75]，超過255是正常的嗎？
    # return img_result, img_confidence, img_data, img_gradient
    return img_result
    
def generate_result_image(img_input, img):
    img_result = np.zeros(img_input.shape, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_result[i][j] = img[i][j].value
    return img_result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--mask')
    parser.add_argument('--output')
    parser.add_argument('--patch_size', type=int, default=3)
    args = parser.parse_args()

    img_input = cv2.imread(args.input, cv2.IMREAD_COLOR) # 3 channel BGR color image
    img_mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE) # 255~240 and 0~20
    img_mask = init_mask(img_mask) # only 255 and 0, 255 == in fillfront
    global patch_size
    patch_size = args.patch_size
    global shape
    shape = img_input.shape
    img = init_image(img_input, img_mask)

    # target_patch_point_idx = find_maxpriority_patch(img)
    # source_patch = find_source_patch(target_patch_point_idx, img)
    # fill_imagedata(img[target_patch_point_idx[0]][target_patch_point_idx[1]], source_patch)
    
    # for iter in range(10):
    iter = 0
    while len(contour_point) != 0:
        print("iter", iter)
        target_patch_point_idx = find_maxpriority_patch(img)
        _ = img[target_patch_point_idx[0], target_patch_point_idx[1]].gradient_vector(2, True)
        _ = img[target_patch_point_idx[0], target_patch_point_idx[1]].normal_direction(True)

        source_patch = find_source_patch(target_patch_point_idx, img)
        fill_imagedata(img[target_patch_point_idx[0]][target_patch_point_idx[1]], source_patch)

        # img_output, img_confidence, img_data, img_gradient = generate_result_image_test(img_input, img, target_patch_point_idx, source_patch) # 單純測試有沒有找到欲填範圍的邊緣
        img_output = generate_result_image_test(img_input, img, target_patch_point_idx, source_patch) # 單純測試有沒有找到欲填範圍的邊緣
        # cv2.imwrite(f"./result_fixcontour/result8_iter{iter}.png", img_output)
        # cv2.imwrite(f"./result_fixcontour/confidence8_iter{iter}.png", img_confidence)
        # cv2.imwrite(f"./result_fixcontour/data8_iter{iter}.png", img_data)
        # cv2.imwrite(f"./result_fixcontour/gradient8_iter{iter}.png", img_gradient)
        cv2.imwrite(f"./result/test2/result10_iter{iter}.png", img_output)
        update_contour_point(img)
        iter += 1
    img_output = generate_result_image(img_input, img)
    
    cv2.imwrite(args.output, img_output)

if __name__ == "__main__":
    main()
