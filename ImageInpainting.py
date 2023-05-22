import cv2
import numpy as np
import argparse

alpha = 255 # normalization factor

# class Image:
#     def __init__(self, pixels, contour):
#         self.pixels = pixels
#         self.contour = contour

class Pixel:
    def __init__(self, r, c, value, is_filled, is_fillfront):
        self.r = r
        self.c = c
        self.value = value # [B, G, R]
        self.confidence = 1 if is_filled else 0
        self.is_filled = is_filled # filled or not
        self.is_fillfront = is_fillfront # in the area to be filled

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
        self.confidence =  confidence_sum / (patch_size**2)
        # print(f"[{self.r}, {self.c}]'s confidence: {self.confidence}")    
        return self.confidence
    
    def compute_data(self):
        """ Compute the data on linear structures around the pixel. """
        norm = self.normal_direction()
        gradient = self.gradient_vector()
        data = np.abs(norm.dot(gradient)) / alpha
        return data

    def compute_patch_priority(self):
        return self.compute_confidence() * self.compute_data()
    
    def gradient_vector(self, k = 2): # per pixel (Sobel)
        max_gradient = -1
        for i in range(self.patch.shape[0]):
            for j in range(self.patch.shape[1]):
                point = self.patch[i][j]
                G = [0 for _ in range(3)]
                for channel in range(3):
                    gr = ((point.neighbors[0][2].value[channel] + k * point.neighbors[1][2].value[channel] + point.neighbors[2][2].value[channel]) - (point.neighbors[0][0].value[channel] + k * point.neighbors[1][0].value[channel] + point.neighbors[2][0].value[channel])) / (k + 2)
                    gc = ((point.neighbors[0][0].value[channel] + k * point.neighbors[0][1].value[channel] + point.neighbors[0][2].value[channel]) - (point.neighbors[2][0].value[channel] + k * point.neighbors[2][1].value[channel] + point.neighbors[2][2].value[channel])) / (k + 2)
                    G[channel] = np.hypot(gr, gc) # sqrt(x*x + y*y)
                gradient = sum(G) / 3
                if gradient > max_gradient:
                    max_gradient = gradient
                    max_gradient_vec = np.array([gc, gr])
        magnitude = np.sqrt(max_gradient_vec.dot(max_gradient_vec))
        # gradient_vec = max_gradient_vec / magnitude * max_gradient # normalize?
        gradient_vec = max_gradient_vec
        return gradient_vec
    
    def normal_direction(self):
        # theta = np.arctan2(prev_point[0]-next_point[0], prev_point[1]-next_point[1])
        # normal = theta - 0.5 * np.pi
        prev_point = [-1, -1]
        for i in range(len(contour_point)):
            point = contour_point[i]
            if contour_point[i] == [self.r, self.c]:
                if prev_point == [-1, -1]:
                    prev_point = contour_point[-1]
                if i == len(contour_point)-1:
                    next_point = contour_point[0]
                else:
                    next_point = contour_point[i+1]
                break
            prev_point = point
        normal = np.array([prev_point[1]-next_point[1], prev_point[0]-next_point[0]])
        magnitude = np.sqrt(normal.dot(normal))
        normal = normal / magnitude * alpha  # normalize?
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
def init_mask(mask_img): # 
    output_image = np.ones_like(mask_img)
    for i in range(mask_img.shape[0]):
        for j in range(mask_img.shape[1]):
            output_image[i, j] = 0 if mask_img[i, j] < 128 else 255
    return output_image

def init_image(img_input, img_mask, patch_size = 3):
    img = np.empty(shape=img_input.shape[0:2], dtype = np.dtype(Pixel))
    first_mask_pixel_xy = [-1, -1]
    global contour_point
    contour_point = []
    for i in range(img_mask.shape[0]):
        for j in range(img_mask.shape[1]): 
            pixel = Pixel(i, j, img_input[i][j], img_mask[i][j] == 0, img_mask[i][j] == 255) 
            img[i][j] = pixel
            if img_mask[i][j] == 255 and first_mask_pixel_xy == [-1, -1]:
                first_mask_pixel_xy = [i, j]

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
                        pixel = Pixel(-1, -1, [-1, -1, -1], -1, -1) # Should not be -1? 算gradient會有問題
                        neighbors[i_patch][j_patch] = pixel
            img[i][j].set_neighbors(neighbors)

    now_x, now_y = first_mask_pixel_xy
    contour_point.append(first_mask_pixel_xy)
    # break_point = False # 找完所有邊緣曲線上的點

    # 為了算邊緣線上點的法向量，找所有邊緣曲線的點放進 contour_point，而且要 contour_point 裏的順序是連著線的
    # 要處理很多個區塊要補的狀況！！！
    '''
    while True: 
        left, right, up, down = max(now_x-patch_size//2, 0), min(now_x+1+patch_size//2, img_mask.shape[0]), max(now_y-patch_size//2, 0), min(now_y+1+patch_size//2, img_mask.shape[1])
        patch = img_mask[left:right, up:down]

        next = False # 可以找下一個邊緣曲線上的點
        for i in range(patch.shape[0]):
            for j in range(patch.shape[1]):
                this_point = [left + i , up + j]
                if is_contour(img_mask, this_point) and this_point not in contour_point:
                    contour_point.append(this_point)
                    now_x, now_y = this_point
                    next = True
                    break
                if len(contour_point) > 30 and [left + i , up + j] == first_mask_pixel_xy: # 找回原本的點了（30 只是隨便設定一個數）
                    break_point = True
            if next:
                break
        
        if break_point:
            break
        # time.sleep(1)
    '''
    while True:
        if len(contour_point) > 1 and [now_x, now_y] == first_mask_pixel_xy: # 找回原本的點了（30 只是隨便設定一個數）
            break
        neighbors = img[now_x][now_y].neighbors
        connected_4 = ([0, 1], [1, 2], [2, 1], [1, 0])
        connected_8 = ([0, 0], [0, 2], [2, 2], [2, 0])
        found = False
        for point in connected_4+connected_8:
            point_coord = [now_x+point[0]-1, now_y+point[1]-1]
            if point_coord[0] < 0 or point_coord[0] >= shape[0] or point_coord[1] < 0 or point_coord[1] >= shape[1]:
                continue
            if is_contour(neighbors[point[0]][point[1]]) and point_coord not in contour_point:
                contour_point.append(point_coord)
                now_x, now_y = point_coord
                found = True
                break
        if not found:
            break
    
    # img = Image(pixels, contour_point)
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

def compute_similarity(target_patch, source_patch):
    # target_patch 中會有很多是待填滿的點，在比較的時候是不是只比較已填或不用填的點們去跟 source_patch 比？
    difference = 0
    for i in range(patch_size):
        for j in range(patch_size):
            if source_patch[i, j].is_filled:
                p1, p2 = target_patch[i, j].value, source_patch[i, j].value # p1, p2 = [B, G, R], [B, G, R]

    return 

def find_source_patch(target_patch_point_idx, img):
    target_patch = img[target_patch_point_idx[0], target_patch_point_idx[1]].patch
    max_similarity = 0
    max_similarity_patch = target_patch
    for row in img:
        for ele in row: # ele is a Pixel
            source_patch = ele.patch
            # 確保 source_patch 裡面每個點都是有顏色的（填滿的）
            similarity = compute_similarity(target_patch, source_patch)
            if  similarity > max_similarity:
                max_similarity_patch = source_patch
                max_similarity = similarity

    return max_similarity_patch

def fill_imagedata(target_patch, source_patch):
    return

def update_confidence(image):
    return

def is_fillfront_empty(img):
    for i in img:
        for pixel in i:
            if not pixel.is_filled:
                return False
    
    return True

def generate_result_image_test(img_input, img, point_idx): # 單純測試有沒有找到欲填範圍的邊緣
    img_result = np.zeros(img_input.shape, dtype=np.uint8)
    # max_magnitude = -1
    # max_data = -1
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if [i, j] in contour_point:
                img_result[i, j] = [255, 0, 0]
                norm = img[i][j].normal_direction()
                gradient = img[i][j].gradient_vector()
                data = img[i][j].compute_data()
                # print(i, j, norm, gradient, data)
                # magnitude = np.sqrt(gradient.dot(gradient))
                # if magnitude > max_magnitude:
                #     max_magnitude = magnitude
                #     max_magnitude_vec = gradient
                # if data > max_data:
                #     max_data = data
                #     max_data_norm = norm
                #     max_data_gradient = gradient
            elif img[i, j].is_fillfront:
                img_result[i, j] = [255, 255, 255]
            else:
                img_result[i, j] = img[i, j].value
    # print(max_magnitude, max_magnitude_vec)
    # print(max_data, max_data_norm, max_data_gradient)
    img_result[point_idx[0], point_idx[1]] = [0, 0, 255]
    # max_data會到 265.30946553035, norm=[114.03946685 228.0789337 ], gradient=[111.75 240.75]，超過255是正常的嗎？
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
    img = init_image(img_input, img_mask, args.patch_size) # patch_size is for compute gradient
    # print(img.shape)

    # 目前完成了找到欲填範圍的邊緣線(演算法有點小漏洞，待修），可以算出邊緣線上點的法向量（np)
    # 而線性結構方向向量（Ip）可以用單元三在教 Sobel 時算 gradient_vector 的部分
    # 但算 np 和 Ip 還沒寫完 （for compute data）

    target_patch_point_idx = find_maxpriority_patch(img)
    # while not is_fillfront_empty(img):
    #     target_patch_point_idx = find_maxpriority_patch(img)
    #     source_patch = find_source_patch(target_patch, img)
    #     fill_imagedata(target_patch, source_patch)
    #     update_confidence(img)
    # img_output = generate_result_image(img)
    img_output = generate_result_image_test(img_input, img, target_patch_point_idx) # 單純測試有沒有找到欲填範圍的邊緣
    cv2.imwrite(args.output, img_output)

if __name__ == "__main__":
    main()
