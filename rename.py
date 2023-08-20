import os
import time
import cv2

# 指定要重命名的文件夹路径和文件名格式
folder_path = 'C:/Users/HZY/Videos/img/'
file_format = '{}.jpg'
#
# 获取文件夹内的所有文件名
file_names = os.listdir(folder_path)
#
# 遍历文件夹内的每个文件，并重命名
for i, old_name in enumerate(file_names):
    # img = cv2.imread(folder_path+old_name)
    # img = cv2.resize(img, (640, 384))
    # cv2.imwrite(folder_path+old_name, img)
    new_name = file_format.format('23430low'+str(i))  # 根据文件名格式生成新的文件名
    os.rename(os.path.join(folder_path, old_name), os.path.join(folder_path, new_name))  # 重命名文件
    print(f'Renamed {old_name} to {new_name}')  # 输出重命名结果
# path1 = "C:/Users/HZY/Pictures/317high_pickarea/images/"        # (1920, 1184)
# path2 = "E:/RobotCenter/dataSet/3.10image/moveroi/"     # (120, 122)
# path3 = "E:/RobotCenter/dataSet/3.10image/deldata/"      # (1920, 1184)
# i = 2
# k = 1
# a = 3
# d = 117
# while d < 215:
#     t1 = time.time()
#     img1 = cv2.imread(path1+str(i)+".jpg")
    # img2 = cv2.imread(path2 + str(a) + ".jpg")
    # print("img.shape", img2.shape[0], img2.shape[1])
    # img3 = img1
    # img3[100+k*50:100+img2.shape[0]+k*50, 100+k*50:100+img2.shape[1]+k*50] = img2
    # cv2.namedWindow("newdarw", cv2.WINDOW_NORMAL)
    # cv2.imshow("newdarw", img3)
    # i += 1
    # # k += 1
    # # if i % 16 == 0:
    # #     k = 1
    # #     a += 1
    # cv2.imwrite(path1 + str(d) + ".jpg", img1)
    # d += 1
    # cv2.waitKey(0)

