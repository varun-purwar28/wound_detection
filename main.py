# main.py

from preprocessing import preprocess
from detection import detect
from cropping import crop
from segmentation import segment
from postprocessing import postprocess
from measurements import measure

def main():
    # Step 1: Preprocessing
    preprocess()

    # Step 2: Detection
    detect()

    # Step 3: Cropping
    crop()

    # Step 4: Segmentation
    segment()

    # Step 5: Postprocessing
    postprocess()

    # Step 6: Measurements
    measure()

if __name__ == "__main__":
    main()
