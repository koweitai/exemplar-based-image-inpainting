import cv2
import numpy as np
import argparse

class Pixel:
    def __init__(self, value, confidence, data, is_filled, is_fillfront):
        self.value = value
        self.confidence = confidence
        self.data = data
        self.is_filled = is_filled
        self.is_fillfront = is_fillfront

    def compute_dataterm(self):
        return

    def compute_priority(self):
        return self.confidence * self.compute_dataterm()

def init_image(img_input):
    img = np.empty(shape=img_input.shape, dtype = np.dtype(Pixel))
    for i in range(img_input.shape[0]):
        for j in range(img_input.shape[1]):
            pixel = Pixel(img_input[i][j], 1, 0, False, False)
            img[i][j] = pixel
    return img

def generate_result_image(img):
    img_result = np.zeros(img.shape, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_result[i][j] = img[i][j].value
    return img_result

def find_maxpriority_patch(fillfront):
    max_priority = 0
    for point in fillfront:
        if point.compute_priority() > max_priority:
            max_priority_point = point

    return max_priority_point

def compute_similarity(target_patch, source_patch):
    return

def find_source_patch(target_patch, img):
    max_similarity = 0
    for source_patch in img:
        if compute_similarity(target_patch, source_patch) > max_similarity:
            max_similarity_patch = source_patch

    return max_similarity_patch



def copy_imagedata(target_patch, source_patch):
    return

def update_fillfromt(image):
    return

def update_confidence(image):
    return

def is_fillfront_empty(image):
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--mask')
    parser.add_argument('--output')
    parser.add_argument('--patch_size')
    args = parser.parse_args()

    img_input = cv2.imread(args.input, cv2.IMREAD_COLOR) # 3 channel BGR color image

    img_mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    print(img_input[0][0])
    # img = init_image(img_input)
    # while not is_fillfront_empty(img):
    #     target_patch = find_maxpriority_patch(img)
    #     source_patch = find_source_patch(target_patch, img)
    #     copy_imagedata(target_patch, source_patch)
    #     update_fillfromt(img)
    #     update_confidence(img)
    # img_output = generate_result_image(img)
    # cv2.imwrite(args.output, img_output)

if __name__ == "__main__":
    main()