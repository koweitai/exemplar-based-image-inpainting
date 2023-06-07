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
        """ Computes confidence of the central pixel of the patch. """
        confidence_sum = sum([ele.confidence for row in self.patch for ele in row if ele.is_filled])
        confidence = confidence_sum / (self.patch.shape[0]*self.patch.shape[1])   
        return confidence
    
    def compute_data(self):
        """ Computes the data on linear structures around the pixel. """
        norm = self.normal_direction()
        gradient = self.gradient_vector()
        data = np.abs(norm.dot(gradient)) / alpha
        self.data = data
        return data

    def compute_patch_priority(self):
        return self.compute_confidence() * self.compute_data()

    def gradient_vector(self, k = 2, printValue = False):
        """ Returns the gradient vector with the highest magnitude in the patch. """
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
        gradient_vec = max_gradient_vec
        self.gradient = max_gradient
        
        if printValue:
            print(f"this patch's gradient: {gradient_vec}")
        
        return gradient_vec

    def normal_direction(self, printValue = False):
        """ Returns the normal vector on the pixel. """
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
    """ Decides whether a pixel is on the contour. """
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
    """ Returns the pixel whose patch has the max priority to be filled. """
    return max(contour_point, key=lambda idx: img[idx[0], idx[1]].compute_patch_priority()) # max_priority_point_idx

def compute_SSIM(target_patch, source_patch):
    """ Returns SSIM between target patch and source patch. """
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
    """ Returns sum of square difference between target patch and source patch. Weighted by confidence value. """
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
    """ Finds the source patch that best fits the target patch. """
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
    """ Fills target patch with source patch. """
    target_patch_pixel.confidence = target_patch_pixel.compute_confidence()
    for i in range(source_patch.shape[0]):
        for j in range(source_patch.shape[1]):
            if (not target_patch_pixel.patch[i][j].is_filled) and source_patch[i][j].is_filled:
                target_patch_pixel.patch[i][j].value = source_patch[i][j].value
                target_patch_pixel.patch[i][j].is_filled = True
                target_patch_pixel.patch[i][j].confidence = target_patch_pixel.confidence
    return

def update_contour_point(img):
    """ Finds a new set of contour points after an iteration. """
    contour_point.clear()
    for i in range(shape[0]):
        for j in range(shape[1]):
            if (not img[i][j].is_filled) and is_contour(img[i][j]) and ([i, j] not in contour_point):
                img[i][j].is_contour = 1
                contour_point.append([i, j])
            else:
                img[i][j].is_contour = 0
    return

def generate_result_image_test(img_input, img, point_idx, source_patch):
    """ During the process, generates current result with source, target and contour information. """
    img_result = np.zeros(img_input.shape, dtype=np.uint8)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if [i, j] in contour_point:
                img_result[i, j] = [255, 0, 0]
            elif not img[i, j].is_filled:
                img_result[i, j] = [255, 255, 255]
            else:
                img_result[i, j] = img[i, j].value
    for i in range(source_patch.shape[0]):
        img_result[source_patch[i][0].r, source_patch[i][0].c] = [0, 255, 0]
        img_result[source_patch[i][source_patch.shape[1]-1].r, source_patch[i][source_patch.shape[1]-1].c] = [0, 255, 0]
    for j in range(source_patch.shape[1]):
        img_result[source_patch[0][j].r, source_patch[0][j].c] = [0, 255, 0]
        img_result[source_patch[source_patch.shape[0]-1][j].r, source_patch[source_patch.shape[0]-1][j].c] = [0, 255, 0]
    img_result[point_idx[0], point_idx[1]] = [0, 0, 255]
    
    return img_result
    
def generate_result_image(img_input, img):
    """ Generates final result image. """
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

        img_output = generate_result_image_test(img_input, img, target_patch_point_idx, source_patch)
        cv2.imwrite(f"./result/result1_iter{iter}.png", img_output)
        update_contour_point(img)
        iter += 1
    img_output = generate_result_image(img_input, img)
    
    cv2.imwrite(args.output, img_output)

if __name__ == "__main__":
    main()