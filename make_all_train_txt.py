import glob

source_img_dir = "/root/autodl-tmp/VMamba/datasets/soda10m/images/train"
target_img_dir = "/root/autodl-tmp/VMamba/datasets/once/images/train"

source_imgs = glob.glob(f"{source_img_dir}/*.jpg")
target_imgs = glob.glob(f"{target_img_dir}/*.jpg")

with open("/root/autodl-tmp/ultralytics/data/all_train.txt", "w") as f:
    for img in source_imgs:
        f.write(img + "\n")
    for img in target_imgs:
        f.write(img + "\n")
print(f"写入图片数: {len(source_imgs) + len(target_imgs)}")