import cv2
import numpy as np
import argparse

def compute_dataterm(point):
    return

def compute_priority(point):
    return point.confidence * compute_dataterm(point)

def find_maxpriority_patch():
    max_priority = 0
    for point in fillfront:
        if compute_priority(point) > max_priority:
            max_priority_point = point

    return max_priority_point

def compute_similarity(target_patch, source_patch):
    return

def find_source_patch(target_patch):
    max_similarity = 0
    for source_patch in image:
        if compute_similarity(target_patch, source_patch) > max_similarity:
            max_similarity_patch = source_patch

    return max_similarity_patch

def copy_imagedata(target_patch, source_patch):
    return

def update_confidence(image):
    return

def is_fillfront_empty(image):
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    args = parser.parse_args()

    img_input = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    while not is_fillfront_empty(img_input):
        target_patch = find_maxpriority_patch()
        source_patch = find_source_patch(target_patch)
        copy_imagedata(target_patch, source_patch)
        update_confidence(img_input)

if __name__ == "__main__":
    main()