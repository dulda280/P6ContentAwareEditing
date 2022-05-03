import os

import keyboard
from tqdm import tqdm
import cv2
from FileManager import *
from ImageProcessing import *
import screeninfo
from Gamma import *
from Downsampling import *

class DataGenerator:
    input_directory = ""
    output_directory = ""

    def __init__(self, input_directory: str, output_directory: str):
        self.input_directory = input_directory
        self.output_directory = output_directory

    def start_data_generator(self, verbose: bool):
        # Class Instantiations
        fileManager = FileManager()
        imageProcessing = ImageProcessing()
        downSampling = Downsampling()

        # Properties for Text Display on images
        fF = cv2.FONT_HERSHEY_PLAIN
        fS = 1
        c = (0, 0, 255)
        t = 2

        # Index Iterator to determine image number
        index = 148

        print("+-----------------------------------+")
        print("| Data Generation Started:")
        print("| Input Directory:", self.input_directory)
        print("| Output Directory:", self.output_directory)
        print("+-----------------------------------+")
        print("| Images in Input Directory:", len(os.listdir(self.input_directory)))
        print("+-----------------------------------+")
        print("Checking filetype integrity...")
        self.directory_filetype_checker()

        for image in os.listdir(self.input_directory):
            loaded_image = cv2.imread(self.input_directory + "/" + str(image), cv2.IMREAD_COLOR)
            loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)

            # Rescaled Images
            rescaled_image = downSampling.rescale_image(loaded_image, 325, 325)
            pr_rescaled_image = downSampling.rescale_image(loaded_image, 325, 325)

            # 0.5 - 1.5

            # Gamma Adjusted Images
            gamma_image_05 = imageProcessing.adjust_gamma(rescaled_image, gamma=0.5)
            pr_gamma_image_05 = imageProcessing.adjust_gamma(rescaled_image, gamma=0.5)
            gamma_image_06 = imageProcessing.adjust_gamma(rescaled_image, gamma=0.6)
            pr_gamma_image_06 = imageProcessing.adjust_gamma(rescaled_image, gamma=0.6)
            gamma_image_07 = imageProcessing.adjust_gamma(rescaled_image, gamma=0.7)
            pr_gamma_image_07 = imageProcessing.adjust_gamma(rescaled_image, gamma=0.7)
            gamma_image_08 = imageProcessing.adjust_gamma(rescaled_image, gamma=0.8)
            pr_gamma_image_08 = imageProcessing.adjust_gamma(rescaled_image, gamma=0.8)
            gamma_image_09 = imageProcessing.adjust_gamma(rescaled_image, gamma=0.9)
            pr_gamma_image_09 = imageProcessing.adjust_gamma(rescaled_image, gamma=0.9)
            gamma_image_11 = imageProcessing.adjust_gamma(rescaled_image, gamma=1.1)
            pr_gamma_image_11 = imageProcessing.adjust_gamma(rescaled_image, gamma=1.1)
            gamma_image_12 = imageProcessing.adjust_gamma(rescaled_image, gamma=1.2)
            pr_gamma_image_12 = imageProcessing.adjust_gamma(rescaled_image, gamma=1.2)
            gamma_image_13 = imageProcessing.adjust_gamma(rescaled_image, gamma=1.3)
            pr_gamma_image_13 = imageProcessing.adjust_gamma(rescaled_image, gamma=1.3)
            gamma_image_14 = imageProcessing.adjust_gamma(rescaled_image, gamma=1.4)
            pr_gamma_image_14 = imageProcessing.adjust_gamma(rescaled_image, gamma=1.4)
            gamma_image_15 = imageProcessing.adjust_gamma(rescaled_image, gamma=1.5)
            pr_gamma_image_15 = imageProcessing.adjust_gamma(rescaled_image, gamma=1.5)

            # Text Display
            cv2.putText(img=pr_rescaled_image, text='Original "A" to Save', org=(0, 12), fontFace=fF, fontScale=fS, color=c,
                        thickness=t)
            cv2.putText(img=pr_gamma_image_05, text='G: 0.5 "Q" to Save', org=(0, 12), fontFace=fF, fontScale=fS,
                        color=c, thickness=t)
            cv2.putText(img=pr_gamma_image_06, text='G: 0.6 "W" to Save', org=(0, 12), fontFace=fF, fontScale=fS,
                        color=c, thickness=t)
            cv2.putText(img=pr_gamma_image_07, text='G: 0.7 "E" to Save', org=(0, 12), fontFace=fF, fontScale=fS,
                        color=c, thickness=t)
            cv2.putText(img=pr_gamma_image_08, text='G: 0.8 "R" to Save', org=(0, 12), fontFace=fF, fontScale=fS,
                        color=c, thickness=t)
            cv2.putText(img=pr_gamma_image_09, text='G: 0.9 "T" to Save', org=(0, 12), fontFace=fF, fontScale=fS,
                        color=c, thickness=t)
            cv2.putText(img=pr_gamma_image_11, text='G: 1.1 "Y" to Save', org=(0, 12), fontFace=fF, fontScale=fS,
                        color=c, thickness=t)
            cv2.putText(img=pr_gamma_image_12, text='G: 1.2 "U" to Save', org=(0, 12), fontFace=fF, fontScale=fS,
                        color=c, thickness=t)
            cv2.putText(img=pr_gamma_image_13, text='G: 1.3 "I" to Save', org=(0, 12), fontFace=fF, fontScale=fS,
                        color=c, thickness=t)
            cv2.putText(img=pr_gamma_image_14, text='G: 1.4 "O" to Save', org=(0, 12), fontFace=fF, fontScale=fS,
                        color=c, thickness=t)
            cv2.putText(img=pr_gamma_image_15, text='G: 1.5 "P" to Save', org=(0, 12), fontFace=fF, fontScale=fS,
                        color=c, thickness=t)

            # Groups all images and displays them horizontally
            concatenated_images_1 = np.concatenate(
                (pr_rescaled_image, pr_gamma_image_05, pr_gamma_image_06, pr_gamma_image_07,
                 pr_gamma_image_08, pr_gamma_image_09), axis=1)

            concatenated_images_2 = np.concatenate(
                (pr_gamma_image_11, pr_gamma_image_12, pr_gamma_image_13, pr_gamma_image_14, pr_gamma_image_15), axis=1)

            # Window and Monitor Properties
            window_name_1 = 'Images 1'
            window_name_2 = 'Images 2'
            screen_id = 1
            screen = screeninfo.get_monitors()[screen_id]
            cv2.namedWindow(window_name_1)
            cv2.namedWindow(window_name_2)
            cv2.moveWindow(window_name_1, screen.x - 1, screen.y - 1)
            cv2.moveWindow(window_name_2, screen.x - 1, screen.y + 355)
            cv2.imshow(window_name_1, concatenated_images_1)
            cv2.imshow(window_name_2, concatenated_images_2)
            cv2.waitKey(100)

            availableKeys = ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"]
            # User Input
            print("Current Reached Index:", index)
            userInput = self.check_user_input()
            if userInput not in availableKeys:
                print("You fucked up, start over")
                self.check_user_input()
            elif userInput == 'q':
                fileManager.save_image_nokeyword(path="Original_Directory", image=rescaled_image, index=index)
                fileManager.save_image_nokeyword(path="Output_Directory", image=gamma_image_05, index=index)
                fileManager.save_text_file(path="Gamma_Values", value="0.5")
                print("Saved image successfully with a gamma of 0.5")
            elif userInput == 'w':
                fileManager.save_image_nokeyword(path="Original_Directory", image=rescaled_image, index=index)
                fileManager.save_image_nokeyword(path="Output_Directory", image=gamma_image_06, index=index)
                fileManager.save_text_file(path="Gamma_Values", value="0.6")
                print("Saved image successfully with a gamma of 0.6")
            elif userInput == 'e':
                fileManager.save_image_nokeyword(path="Original_Directory", image=rescaled_image, index=index)
                fileManager.save_image_nokeyword(path="Output_Directory", image=gamma_image_07, index=index)
                fileManager.save_text_file(path="Gamma_Values", value="0.7")
                print("Saved image successfully with a gamma of 0.7")
            elif userInput == 'r':
                fileManager.save_image_nokeyword(path="Original_Directory", image=rescaled_image, index=index)
                fileManager.save_image_nokeyword(path="Output_Directory", image=gamma_image_08, index=index)
                fileManager.save_text_file(path="Gamma_Values", value="0.8")
                print("Saved image successfully with a gamma of 0.8")
            elif userInput == 't':
                fileManager.save_image_nokeyword(path="Original_Directory", image=rescaled_image, index=index)
                fileManager.save_image_nokeyword(path="Output_Directory", image=gamma_image_09, index=index)
                fileManager.save_text_file(path="Gamma_Values", value="0.9")
                print("Saved image successfully with a gamma of 0.9")
            elif userInput == 'y':
                fileManager.save_image_nokeyword(path="Original_Directory", image=rescaled_image, index=index)
                fileManager.save_image_nokeyword(path="Output_Directory", image=gamma_image_11, index=index)
                fileManager.save_text_file(path="Gamma_Values", value="1.1")
                print("Saved image successfully with a gamma of 1.1")
            elif userInput == 'u':
                fileManager.save_image_nokeyword(path="Original_Directory", image=rescaled_image, index=index)
                fileManager.save_image_nokeyword(path="Output_Directory", image=gamma_image_12, index=index)
                fileManager.save_text_file(path="Gamma_Values", value="1.2")
                print("Saved image successfully with a gamma of 1.2")
            elif userInput == 'i':
                fileManager.save_image_nokeyword(path="Original_Directory", image=rescaled_image, index=index)
                fileManager.save_image_nokeyword(path="Output_Directory", image=gamma_image_13, index=index)
                fileManager.save_text_file(path="Gamma_Values", value="1.3")
                print("Saved image successfully with a gamma of 1.3")
            elif userInput == 'o':
                fileManager.save_image_nokeyword(path="Original_Directory", image=rescaled_image, index=index)
                fileManager.save_image_nokeyword(path="Output_Directory", image=gamma_image_14, index=index)
                fileManager.save_text_file(path="Gamma_Values", value="1.4")
                print("Saved image successfully with a gamma of 1.4")
            elif userInput == 'p':
                fileManager.save_image_nokeyword(path="Original_Directory", image=rescaled_image, index=index)
                fileManager.save_image_nokeyword(path="Output_Directory", image=gamma_image_15, index=index)
                fileManager.save_text_file(path="Gamma_Values", value="1.5")
                print("Saved image successfully with a gamma of 1.5")
            elif userInput == 'a':
                fileManager.save_image_nokeyword(path="Original_Directory", image=rescaled_image, index=index)
                fileManager.save_image_nokeyword(path="Output_Directory", image=rescaled_image, index=index)
                fileManager.save_text_file(path="Gamma_Values", value="1.0")
                print("Saved image successfully with a gamma of 1.0 *Original*")

            cv2.destroyAllWindows()
            index += 1

    def directory_filetype_checker(self):
        for image in tqdm(os.listdir(self.input_directory)):

            if image.endswith(".jpg") or image.endswith(".jpeg") or image.endswith(".png") or image.endswith(".JPG"):
                continue
            else:
                raise ValueError("Image is not of filetype 'jpg' or 'png' stopping generator.")

        # If checker verifies integrity of files successfully
        print("Filetype Checker: All images satisfy filetype requirements")

    def check_user_input(self):
        userInput = input(
            "Keypress Options: | Q: 0.5 | W: 0.6 | E: 0.7 | R: 0.8 | T: 0.9 | Y: 1.1 | U: 1.2 | I: 1.3 | O: 1.4 | P: 1.5 |" + "\n" + "Choose Picture To Save:")
        return userInput


if __name__ == '__main__':
    dataGenerator = DataGenerator(input_directory="Input_Directory_Landscape", output_directory="Output_Directory")
    dataGenerator.start_data_generator(True)
