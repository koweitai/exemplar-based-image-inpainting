import cv2
import numpy as np
import argparse

# class Image:
#     def __init__(self, pixels, contour):
#         self.pixels = pixels
#         self.contour = contour

class Pixel:
    def __init__(self, r, c, value, patch, is_filled, is_fillfront):
        self.r = r
        self.c = c
        self.value = value # [B, G, R]
        self.patch = patch
        self.confidence = 0 if is_fillfront else 1
        # self.data = self.compute_data()
        self.is_filled = is_filled # filled or not
        self.is_fillfront = is_fillfront # in the area to be filled

    def compute_confidence(self):
        """ Compute confidence of the central pixel of the patch. """
        return self.confidence
    
    def compute_data(self, contour):
        """ Compute the data on linear structures around the pixel. """
        # normalization factor
        alpha = 255
        norm = self.normal_direction(contour)

        gradient_vec = self.gradient_magnitude()
        data = np.sqrt(norm.dot(gradient_vec)) / alpha
        return data

    def compute_priority(self, contour):
        return self.compute_confidence() * self.compute_data(contour)
    
    def gradient_magnitude(self, k = 2): # per pixel (Sobel)
        max_gradient = -1
        for point in self.patch:
            gr = ((point.patch[0][2] + k * point.patch[1][2] + point.patch[2][2]) - (point.patch[0][0] + k * point.patch[1][0] + point.patch[2][0])) / (k + 2)
            gc = ((point.patch[0][0] + k * point.patch[0][1] + point.patch[0][2]) - (point.patch[2][0] + k * point.patch[2][1] + point.patch[2][2])) / (k + 2)
            G = np.hypot(gr, gc) # sqrt(x*x + y*y)
            if G > max_gradient:
                max_gradient = G
                max_gradient_vec = np.array([gc, gr])
        magnitude = np.sqrt(max_gradient_vec.dot(max_gradient_vec))
        gradient_vec = max_gradient_vec / magnitude * max_gradient
        return gradient_vec
    
    def normal_direction(self, contour):
        # theta = np.arctan2(prev_point[0]-next_point[0], prev_point[1]-next_point[1])
        # normal = theta - 0.5 * np.pi
        prev_point = [-1, -1]
        for i in range(len(contour)):
            point = contour[i]
            if contour[i] == [self.r, self.c]:
                if prev_point == [-1, -1]:
                    prev_point = contour[-1]
                if i == len(contour)-1:
                    next_point = contour[0]
                else:
                    next_point = contour[i+1]
                break
            prev_point = point
        normal = np.array([prev_point[1]-next_point[1], prev_point[0]-next_point[0]])
        magnitude = np.sqrt(normal.dot(normal))
        normal = normal / magnitude
        return normal

# Tool function
def is_contour(img, idx):
    x, y = idx
    if img[x, y] == 0: # 不是要填滿的區域
        return False
    if x-1 < 0 or x+1 >= img.shape[0] or y-1 < 0 or y+1 >= img.shape[1]: # 該要填滿點在圖的邊邊，視作邊緣
        return True
    if img[x-1, y] == 0 or img[x, y-1] == 0 or img[x+1, y] == 0 or img[x, y+1] == 0: # 上下左右有不是要填滿的區域
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
    contour_point = []
    for i in range(img_mask.shape[0]):
        for j in range(img_mask.shape[1]):
            patch = img_input[max(i-patch_size//2, 0):min(i+1+patch_size//2, img_mask.shape[0]), max(j-patch_size//2, 0):min(j+1+patch_size//2, img_mask.shape[1])]
            pixel = Pixel(i, j, img_input[i][j], patch, False, img_mask[i][j] == 255) 
            img[i][j] = pixel
            if img_mask[i][j] == 255 and first_mask_pixel_xy == [-1, -1]:
                first_mask_pixel_xy = [i, j]

    now_x, now_y = first_mask_pixel_xy
    contour_point.append(first_mask_pixel_xy)
    break_point = False # 找完所有邊緣曲線上的點

    # 為了算邊緣線上點的法向量，找所有邊緣曲線的點放進 contour_point，而且要 contour_point 裏的順序是連著線的
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
    
    # img = Image(pixels, contour_point)
    return img, contour_point

def find_maxpriority_patch(fillfront):
    max_priority = 0
    for point in fillfront:
        if point.is_fillfront and point.compute_priority() > max_priority:
            max_priority_point = point

    return max_priority_point

def compute_similarity(target_patch, source_patch):
    # target_patch 中會有很多是待填滿的點，在比較的時候是不是只比較已填或不用填的點們去跟 source_patch 比？
    return

def find_source_patch(target_patch, img, patch_size):
    max_similarity = 0
    max_similarity_patch = target_patch
    for i in img:
        for j in i:
            source_patch = img[i:i+patch_size, j:j+patch_size]
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
            if pixel.is_fillfront and not pixel.is_filled:
                return False
    
    return True

def generate_result_image_test(img_input, img, contour): # 單純測試有沒有找到欲填範圍的邊緣
    img_result = np.zeros(img_input.shape, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if [i, j] in contour:
                img_result[i, j] = [255, 0, 0]
                print(i, j, img[i][j].normal_direction(contour))
            elif img[i, j].is_fillfront:
                img_result[i, j] = [255, 255, 255]
            else:
                img_result[i, j] = img[i, j].value
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
    img, contour_point = init_image(img_input, img_mask, args.patch_size) # patch_size is for compute gradient
    print(img.shape)
    img_output = generate_result_image_test(img_input, img, contour_point) # 單純測試有沒有找到欲填範圍的邊緣


    # 目前完成了找到欲填範圍的邊緣線(演算法有點小漏洞，待修），可以算出邊緣線上點的法向量（np)
    # 而線性結構方向向量（Ip）可以用單元三在教 Sobel 時算 gradient_magnitude 的部分
    # 但算 np 和 Ip 還沒寫完 （for compute data）


    # while not is_fillfront_empty(img):
    #     target_patch = find_maxpriority_patch(img)
    #     source_patch = find_source_patch(target_patch, img, args.patch_size)
    #     fill_imagedata(target_patch, source_patch)
    #     update_confidence(img)
    # img_output = generate_result_image(img)
    cv2.imwrite(args.output, img_output)

if __name__ == "__main__":
    main()
