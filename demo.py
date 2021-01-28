# coding:utf-8

import os
import ocr
import time
import shutil
import numpy as np
from PIL import Image
from glob import glob
image_files = glob('./test_images/*.*')

#######################################################
# 项目测试入口
# 将测试用的身份证图片放入test_images目录后，运行
# python demo.py
# 识别结果会保存到test_result目录。
#######################################################
if __name__ == '__main__':
    result_dir = './test_result'
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)

    with open(os.path.join(result_dir,"output.txt"),"w") as wp:
        for image_file in sorted(image_files):
            imageName = image_file[image_file.rindex("/") + 1:]
            image = np.array(Image.open(image_file).convert('RGB'))
            t = time.time()
            result, image_framed = ocr.model(image)
            output_file = os.path.join(result_dir, image_file.split('/')[-1])
            Image.fromarray(image_framed).save(output_file)
            print("Mission complete, it took {:.3f}s".format(time.time() - t))
            print("\nRecognition Result:\n")
            wp.write("-------------------------\n")
            wp.write(imageName+"\n")
            for key in result:
                print(result[key][1])
                wp.write(result[key][1]+"\n")
