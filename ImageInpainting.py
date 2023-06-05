import cv2
import numpy as np
import argparse
from skimage.metrics import structural_similarity as ssim

alpha = 255 # normalization factor
block_types = [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
                [[0, 1, 0], [0, 1, 0], [0, 1, 0]],
                [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
                [[0, 1, 0], [0, 1, 0], [1, 0, 0]],
                [[0, 1, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
                [[0, 0, 0], [0, 1, 1], [0, 1, 0]],
                [[0, 1, 0], [0, 1, 0], [1, 0, 0]],
                [[0, 1, 0], [0, 1, 0], [0, 0, 1]],
                [[1, 0, 0], [0, 1, 0], [0, 1, 0]],
                [[0, 0, 1], [0, 1, 0], [0, 1, 0]],
                [[0, 0, 1], [1, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [1, 1, 0], [0, 0, 1]],
                [[1, 0, 0], [0, 1, 1], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 1], [1, 0, 0]],
                [[1, 1, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 1], [0, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 0], [1, 1, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 1, 1]],
                [[1, 0, 0], [1, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [1, 1, 0], [1, 0, 0]],
                [[0, 0, 1], [0, 1, 1], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 1], [0, 0, 1]],
                [[1, 0, 1], [0, 1, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 1, 0], [1, 0, 1]],
                [[1, 0, 0], [0, 1, 0], [1, 0, 0]],
                [[0, 0, 1], [0, 1, 0], [0, 0, 1]]]
normal_types = [np.array([180.3122292, -180.3122292]),
                np.array([180.3122292, 180.3122292]),
                np.array([0., 255.]),
                np.array([255., 0.]),
                np.array([-114.03946685, -228.0789337]),
                np.array([114.03946685, -228.0789337]),
                np.array([180.3122292, -180.3122292]),
                np.array([-180.3122292, -180.3122292]),
                np.array([-114.03946685, -228.0789337]),
                np.array([114.03946685, -228.0789337]),
                np.array([114.03946685, -228.0789337]),
                np.array([-114.03946685, -228.0789337]),
                np.array([-228.0789337 ,-114.03946685]),
                np.array([228.0789337 ,-114.03946685]),
                np.array([228.0789337 ,-114.03946685]),
                np.array([-255., 0.]),
                np.array([255., 0.]),
                np.array([255., 0.]),
                np.array([255., 0.]),
                np.array([255., 0.]),
                np.array([0., -255.]),
                np.array([0., -255.]),
                np.array([0., -255.]),
                np.array([0., -255.]),
                np.array([255., 0.]),
                np.array([255., 0.]),
                np.array([0., -255.]),
                np.array([0., -255.])
                ]

class Pixel:
    def __init__(self, r, c, value, is_filled):
        self.r = r
        self.c = c
        self.value = value # [B, G, R]
        self.confidence = 1 if is_filled else 0
        self.is_filled = is_filled # filled or not
        self.is_contour = 0
        self.data = 0
        self.gradient = 0

    def set_patch(self, patch):
        self.patch = patch

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors # 九宮格的value

    def compute_confidence(self):
        """ Compute confidence of the central pixel of the patch. """
        confidence_sum = sum([ele.confidence for row in self.patch for ele in row if ele.is_filled])
        confidence = confidence_sum / (self.patch.shape[0]*self.patch.shape[1])
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
    
    '''
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
    '''

    def gradient_vector(self, k = 2, printValue = False):
        values = [np.empty(self.patch.shape) for _ in range(3)] # values.shape is [3, patch.shape[0], patch.shape[1]]
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
        gr_sum = sum(gr) / 3 # gr_sum.shape is [patch.shape[0], patch.shape[1]]]
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
        # magnitude = np.sqrt(max_gradient_vec.dot(max_gradient_vec))
        # self.gradient = magnitude
        gradient_vec = max_gradient_vec
        self.gradient = max_gradient
        if printValue:
            print(f"this patch's gradient: {gradient_vec}")
        return gradient_vec
    
    '''
    def gradient_vector_self(self, k = 2): # per pixel (Sobel)
        values = [np.empty(self.patch.shape) for _ in range(3)] # value.shape = [3, patch.shape[0], patch.shape[1]]
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
        gr_sum = sum(gr) / 3 # gr_sum.shape = [patch.shape[0], patch.shape[1]]]
        gc_sum = sum(gc) / 3

        if not (np.isnan(gr_sum[self.patch.shape[0]//2, self.patch.shape[1]//2])) and not (np.isnan(gc_sum[self.patch.shape[0]//2, self.patch.shape[1]//2])):
            gradient = np.sqrt(gr_sum[self.patch.shape[0]//2, self.patch.shape[1]//2]**2 + gc_sum[self.patch.shape[0]//2, self.patch.shape[1]//2]**2)
            gradient_vec = np.array([gc_sum[self.patch.shape[0]//2, self.patch.shape[1]//2], gr_sum[self.patch.shape[0]//2, self.patch.shape[1]//2]])
        else:
            gradient = -1
            gradient_vec = np.array([0, 0])
     
        # magnitude = np.sqrt(gradient_vec.dot(gradient_vec))
        self.gradient = gradient
        return gradient_vec
    '''

    '''
    def normal_direction_old(self, printValue = False):
        cur_point = next((point for point in contour_point if point == [self.r, self.c]), None)
        
        prev_point = [-1, -1]
        next_point = [-1, -1]
        for point in contour_point:
            if point == cur_point:
                continue
            dist = (point[0]-cur_point[0])**2 + (point[1]-cur_point[1])**2
            # if dist <= 2:
            if dist <=3 and dist >= 1:
                if prev_point == [-1, -1]:
                    prev_point = point
                elif next_point == [-1, -1]:
                    next_point = point
                    break
        
        normal = np.array([(-1)*(prev_point[1]-next_point[1]), prev_point[0]-next_point[0]])
        if printValue:
            print(normal)
        magnitude = np.sqrt(normal.dot(normal))
        normal = normal / magnitude * alpha  # normalize?
        if printValue:
            print(prev_point, cur_point, next_point)
            print(f"this patch's normal: {normal}")
        if printValue:
            return normal, prev_point, next_point
        else:
            return normal
    '''

    def normal_direction(self, printValue = False):
        block = [[self.neighbors[i][j].is_contour for j in range(3)] for i in range(3)]
        
        type = -1
        for i in range(len(block_types)):
            if np.array_equal(np.logical_and(block_types[i], block), block_types[i]):
                type = i
                break
        normal = normal_types[type]
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
    return np.where(mask_img < 128, 0, 255)

def init_image(img_input, img_mask):
    img = np.empty(shape=img_input.shape[0:2], dtype = np.dtype(Pixel))
    global contour_point
    contour_point = []
    for i in range(img_mask.shape[0]):
        for j in range(img_mask.shape[1]): 
            pixel = Pixel(i, j, img_input[i][j], img_mask[i][j] == 0) 
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
                        pixel = Pixel(-1, -1, [-1, -1, -1], -1)
                        neighbors[i_patch][j_patch] = pixel
            img[i][j].set_neighbors(neighbors)

    update_contour_point(img)
    
    return img

def find_maxpriority_patch(img):
    return max(contour_point, key=lambda idx: img[idx[0], idx[1]].compute_patch_priority()) # max_priority_point_idx

def compute_SSIM(target_patch, source_patch):
    min_SSIM = -1 # if source_patch 不是填滿的

    if source_patch.shape != target_patch.shape:
        return min_SSIM
    
    img_target = np.zeros([patch_size, patch_size, 3], dtype=np.uint8)
    img_source = np.zeros([patch_size, patch_size, 3], dtype=np.uint8)

    # source_patch 要是滿的，再看 target_patch 有填的點
    for i in range(target_patch.shape[0]):
        for j in range(target_patch.shape[1]):
            '''
            if not source_patch[i, j].is_filled:
                return min_SSIM
            img_source[i, j] = source_patch[i, j].value
            img_target[i, j] = target_patch[i, j].value if target_patch[i, j].is_filled else source_patch[i, j].value
            '''
            if not source_patch[i, j].is_filled:
                return min_SSIM
            if target_patch[i, j].is_filled:
                img_source[i, j] = source_patch[i, j].value
                img_target[i, j] = target_patch[i, j].value
    ssim_value = ssim(img_target, img_source, multichannel=True, win_size=patch_size, channel_axis=2)

    return ssim_value

def compute_sumOfSquare(target_patch, source_patch):
    max_diff = float('inf') # if source_patch 不是填滿的

    if source_patch.shape != target_patch.shape:
        return max_diff

    # source_patch 要是滿的，再看 target_patch 有填的點
    diff = 0
    for i in range(target_patch.shape[0]):
        for j in range(target_patch.shape[1]):
            if not source_patch[i, j].is_filled:
                return max_diff
            if target_patch[i, j].is_filled:
                diff_here = np.array(source_patch[i, j].value.astype(np.uint32) - target_patch[i, j].value.astype(np.uint32))
                diff_here = np.square(diff_here)
                diff += np.sqrt(diff_here.sum()) / source_patch[i, j].confidence / target_patch[i, j].confidence
    return diff

def find_source_patch(target_patch_point_idx, img):
    target_patch = img[target_patch_point_idx[0], target_patch_point_idx[1]].patch
    min_diff = float('inf')
    min_diff_patch = target_patch
    for ele in img.flatten(): # ele is a Pixel
        if not (ele.r == target_patch_point_idx[0] and ele.c == target_patch_point_idx[1]):
            source_patch = ele.patch
            SSIM_value = (1-compute_SSIM(target_patch, source_patch))*5000
            sumOfSquare_value = min(500, (compute_sumOfSquare(target_patch, source_patch)))
            difference = SSIM_value + sumOfSquare_value
            if  difference < min_diff:
                min_diff_patch = source_patch
                min_diff = difference
    
    print(f"(1-SSIM)*5000: {(1-compute_SSIM(target_patch, min_diff_patch))*5000}")
    print(f"sum of Square: {min(500, (compute_sumOfSquare(target_patch, min_diff_patch)))}")
    return min_diff_patch

def fill_imagedata(target_patch_pixel, source_patch):
    target_patch_pixel.confidence = target_patch_pixel.compute_confidence()
    for i in range(source_patch.shape[0]):
        for j in range(source_patch.shape[1]):
            if (not target_patch_pixel.patch[i][j].is_filled) and source_patch[i][j].is_filled:
                target_patch_pixel.patch[i][j].value = source_patch[i][j].value
                target_patch_pixel.patch[i][j].is_filled = True
                target_patch_pixel.patch[i][j].confidence = target_patch_pixel.confidence
    return

def update_contour_point(img):
    contour_point.clear()
    for i in range(shape[0]):
        for j in range(shape[1]):
            if (not img[i][j].is_filled) and is_contour(img[i][j]) and ([i, j] not in contour_point):
                img[i][j].is_contour = 1
                contour_point.append([i, j])
            else:
                img[i][j].is_contour = 0
    return

def generate_result_image_test(img_input, img, point_idx, source_patch): # 單純測試有沒有找到欲填範圍的邊緣
    _ = img[point_idx[0], point_idx[1]].gradient_vector(2, True)
    _ = img[point_idx[0], point_idx[1]].normal_direction(True)
    
    img_result = np.zeros(img_input.shape, dtype=np.uint8)
    # img_confidence = np.zeros(img_input.shape[:-1], dtype=np.uint8)
    # img_data = np.zeros(img_input.shape[:-1], dtype=np.uint8)
    # img_gradient = np.zeros(img_input.shape[:-1], dtype=np.uint8)
    # max_magnitude = -1
    # max_data = -1
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # img_confidence[i, j] = int(img[i, j].confidence * 255)
            # img_data[i, j] = 0 if img[i, j].is_filled else np.clip(img[i, j].data, 0, 255)
            # img_gradient[i, j] = 0 if img[i, j].is_filled else np.clip(img[i, j].gradient, 0, 255)
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
    # img_result[prev_point[0], prev_point[1]] = [0, 255, 0]
    # img_result[next_point[0], next_point[1]] = [0, 255, 0]
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
    
    iter = 0
    while len(contour_point) != 0:
        print("iter", iter)
        target_patch_point_idx = find_maxpriority_patch(img)

        source_patch = find_source_patch(target_patch_point_idx, img)
        fill_imagedata(img[target_patch_point_idx[0]][target_patch_point_idx[1]], source_patch)

        # img_output, img_confidence, img_data, img_gradient = generate_result_image_test(img_input, img, target_patch_point_idx, source_patch) # 單純測試有沒有找到欲填範圍的邊緣
        img_output = generate_result_image_test(img_input, img, target_patch_point_idx, source_patch) # 單純測試有沒有找到欲填範圍的邊緣
        cv2.imwrite(f"./result/result13/result13_iter{iter}.png", img_output)
        # cv2.imwrite(f"./result/test4/confidence10_iter{iter}.png", img_confidence)
        # cv2.imwrite(f"./result/test4/data10_iter{iter}.png", img_data)
        # cv2.imwrite(f"./result/test4/gradient10_iter{iter}.png", img_gradient)
        update_contour_point(img)
        iter += 1
    img_output = generate_result_image(img_input, img)
    
    cv2.imwrite(args.output, img_output)

if __name__ == "__main__":
    main()