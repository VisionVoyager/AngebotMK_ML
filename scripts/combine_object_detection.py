from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import pipeline


def crop_bbx_on_img(img_path, bbx_info):
    img = Image.open(img_path)
    crops = []
    for bb in bbx_info:
        box = bb["box"]
        x, y, w, h = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        temp_bbox = [x, y, x + w, y + h]
        box = [x, y, x + w, y + h]
        if box[2] > img.width:
            temp_bbox[2] = img.width
        if box[3] > img.height:
            temp_bbox[3] = img.height
        crop = img.crop(temp_bbox)

        crops.append(crop)

    return crops


def draw_bbx_on_in_mem_img(img, bbx_info):
    draw = ImageDraw.Draw(img)
    for bb in bbx_info:
        label = bb["label"]
        score = bb["score"]
        box = bb["box"]
        x, y, w, h = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        draw.rectangle((x, y, w, h), outline="red", width=1)
        draw.text((x, y), f"{label} Score: {score}", fill="black")

    return img


def detect_products(img, model_path):
    obj_detector = pipeline("object-detection", model=model_path)
    bbx_info = obj_detector(img)
    return bbx_info


def detect_product_details(img, model_path):
    obj_detector = pipeline("object-detection", model=model_path)
    bbx_info = obj_detector(img)
    return bbx_info


if __name__ == "__main__":

    img_path = "../mock_data/1149219f-letok-01-2023-web-1_17.png"
    product_model_path = "../model_data/300_product_area_only_epoch_run"
    product_details_model_path = "../model_data/300_epochs_subimages"
    img = Image.open(img_path)
    bbx_info = detect_products(img, product_model_path)
    crops = crop_bbx_on_img(img_path, bbx_info)
    for i, image in tqdm(enumerate(crops), total=len(crops)):
        bbx_info = detect_product_details(image, product_details_model_path)
        img = draw_bbx_on_in_mem_img(image, bbx_info)
        img.save(f"output{i}.png")
