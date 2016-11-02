import numpy as np
import cv2
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, help='input text file path', default='out.txt')

    args = parser.parse_args()

    img = np.loadtxt(args.in_file, dtype=int)
    img = img.astype(np.uint8)
    
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
