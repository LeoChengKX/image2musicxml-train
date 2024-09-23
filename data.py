from PIL import Image
import os
import tqdm
import numpy as np
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

abs_path = "D:\Programing\python\Transformers\image2musicxml"
path = os.path.join(abs_path, "dataset")
image_path = os.path.join(path, "Images")
xml_path = os.path.join(path, "MusicXML")
processed_path = os.path.join(path, "Processed")
measure_path = os.path.join(path, "Measure")

START_WITH = "Bartok - Solo Violin Sonata"
OUTPUT_FILE_NAME = f"{START_WITH} splits"
image_files = [f for f in os.listdir(image_path) if f.endswith((".png", ".jpg")) and f.startswith(START_WITH)]

def crop_data():
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    for name in os.listdir(image_path)[:5]:
        if name.endswith((".png")):
            file_path = os.path.join(image_path, name)
            with Image.open(file_path) as img:
                w, h = img.size
                print(name)
                grayscale_img = img.convert("L")
            
                # Apply a threshold to identify black pixels (0 = black, 255 = white)
                threshold = 1  # Any pixel brighter than black will be considered non-black
                binary_img = grayscale_img.point(lambda p: p > threshold and 255)

                # Get the bounding box of the non-black region
                bbox = binary_img.getbbox()

                # If bbox is None, the image is fully black, so no cropping needed
                if bbox:
                    cropped_img = img.crop(bbox)
                    save_path = os.path.join(processed_path, name)
                    cropped_img.save(save_path)

                # img = img.crop((0, 0, w, h-2500))
                # img.save(os.path.join(processed_path, name))


def correct_data_name(file_name: str):
    temp = file_name.split(" - ")
    for i in range(len(temp)):
        if temp[i] == 'Full score':
            return " - ".join(temp[:i])
    
    raise ValueError


def extract_measure_number():
    measures = {}

    if not os.path.exists(measure_path):
        os.makedirs(measure_path)
    f = open(OUTPUT_FILE_NAME, mode='w')

    for name in tqdm.tqdm(image_files, desc="Processing"):
        if name.endswith((".png")):
            file_path = os.path.join(image_path, name)
            with Image.open(file_path).convert("RGB") as img:
                w, h = img.size
                # img = img.crop((200, 250, w-2000, h-2800))
                img = img.crop((0, 250, w-1500, h-2800))
                
                img = np.array(img)
                # Convert RGB to BGR
                img = img[:, :, ::-1].copy()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Performing OTSU threshold
                ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
                rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))

                # Applying dilation on the threshold image
                dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

                # Finding contours
                contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, 
                                                                cv2.CHAIN_APPROX_NONE)

                # Creating a copy of image
                im2 = img.copy()

                possible_meausres = []
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # Drawing a rectangle on copied image
                    rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Cropping the text block for giving input to OCR
                    cropped = im2[y:y + h, x:x + w]
                    # Apply OCR on the cropped image
                    text = pytesseract.image_to_string(cropped, config="--psm 7 digits -c tessedit_char_whitelist=0123456789 --dpi 600")
                    if text.strip().isnumeric():
                        possible_meausres.append(int(text))

                measures[name] = possible_meausres
                if 1 in possible_meausres:
                    possible_meausres.remove(1)

                f.write(name + " ")
                if len(possible_meausres) == 0:
                    f.write("1 ")
                elif len(possible_meausres) == 1:
                    f.write(str(possible_meausres[0]) + " ")
                else:
                    f.write(str(possible_meausres))
                f.write("\n")
                cv2.imwrite(os.path.join(measure_path, name), im2)
    f.close()
    return measures


def save_measure_split(measures: dict):
    with open("measure_split.txt", mode='w') as f:
        for key in measures.keys():
            f.write(key + " ")
            if len(measures[key]) == 0:
                f.write("0 ")
            elif len(measures[key]) == 1:
                f.write(str(measures[key][0]) + " ")
            else:
                f.write(str(measures[key]))
            f.write("\n")
    

if __name__ == '__main__':
    extract_measure_number()