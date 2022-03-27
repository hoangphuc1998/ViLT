import json
import os
import pandas as pd
import pyarrow as pa
import random

from tqdm import tqdm
from glob import glob
from collections import defaultdict
from sklearn.model_selection import train_test_split

def path2rest(path, iid2captions, iid2split):
    name = path.split("/")[-1]
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]
    split = iid2split[name]
    return [binary, captions, name, split]

def make_arrow(root, dataset_root):
    with open(f"{root}/train.json", "r") as f:
        train_annotations = json.load(f)
    with open(f"{root}/val.json", "r") as f:
        val_annotations = json.load(f)
    val_images = [x['file_name'] for x in val_annotations["images"]]
    val_images, test_images = train_test_split(val_images, test_size=5000)
    _, val_images = train_test_split(val_images, test_size=500)
    iid2captions = defaultdict(list)
    iid2split = dict()
    iid2filename = dict()

    for image_info in train_annotations["images"]:
        iid2filename[image_info["id"]] = image_info["file_name"]
        iid2split[image_info["file_name"]] = "train"
    for image_info in val_annotations["images"]:
        iid2filename[image_info["id"]] = image_info["file_name"]
        if image_info["file_name"] in val_images:
            iid2split[image_info["file_name"]] = "val"
        elif image_info["file_name"] in test_images:
            iid2split[image_info["file_name"]] = "test"
        else:
            iid2split[image_info["file_name"]] = "train"

    for annotation in train_annotations["annotations"] + val_annotations["annotations"]:
        filename = iid2filename[annotation["image_id"]]
        iid2captions[filename].append(annotation["caption"])
    
    paths = list(glob(f"{dataset_root}/train/*.jpg")) + list(glob(f"{dataset_root}/val/*.jpg"))
    random.shuffle(paths)
    caption_paths = [path for path in paths if path.split("/")[-1] in iid2captions]
    if len(paths) == len(caption_paths):
        print("all images have caption annotations")
    else:
        print("not all images have caption annotations")
    print(
        len(paths), len(caption_paths), len(iid2captions),
    )

    bs = [path2rest(path, iid2captions, iid2split) for path in tqdm(caption_paths)]

    for split in ["train", "val", "test"]:
        batches = [b for b in bs if b[-1] == split]

        dataframe = pd.DataFrame(
            batches, columns=["image", "caption", "image_id", "split"],
        )

        table = pa.Table.from_pandas(dataframe)
        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(
            f"{dataset_root}/vizwiz_{split}.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)