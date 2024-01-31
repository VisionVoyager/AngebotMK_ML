import json

from datasets import Dataset, load_dataset
from PIL import Image, ImageDraw


def transform_result(result, img_id, exclude_classes=[]):
    annotations = [
        r
        for r in result["annotations"]
        if r["image_id"] == img_id
        if r["category_id"] not in exclude_classes
    ]
    new_result = {
        "image_id": img_id,
        "image": Image.open(data_path + "/" + result["images"][img_id]["file_name"]),
        "width": result["images"][img_id]["width"],
        "height": result["images"][img_id]["height"],
        "objects": {
            "id": [a["id"] for a in annotations],
            "area": [a["area"] for a in annotations],
            "bbox": [a["bbox"] for a in annotations],
            "category": [a["category_id"] for a in annotations],
        },
    }

    return new_result


if __name__ == "__main__":
    data_path = "data"

    with open(data_path + "/result.json", "r") as f:
        angebot = json.load(f)
    rando_str = 45
    angebot_dataset = Dataset.from_list(
        [transform_result(angebot, i, []) for i in range(len(angebot["images"]))]
    )

    print(angebot_dataset[0])
