import os
import cv2
prefix="face-"
root_path = "kagglehub/datasets/jeffheaton/glasses-or-no-glasses/versions/2/"
images = root_path + "faces-spring-2020/faces-spring-2020"
#resizing
dirpath = "resized_dataset"
os.makedirs(dirpath,exist_ok=True)
for i in range(684,5001):
    img_name = prefix + f"{i}.png"
    img = cv2.imread(os.path.join(images,img_name))
    resized_image = cv2.resize(src=img,dsize=(64,64),interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(dirpath,img_name),resized_image)
    print("resized image: ",i)

print("done")
