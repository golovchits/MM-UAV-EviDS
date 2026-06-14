#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import json
import os
import pickle
from loguru import logger
from pathlib import Path

import cv2
cv2.setNumThreads(0)  # prevent TBB deadlock with CUDA in main-process workers=0
import numpy as np
from pycocotools.coco import COCO

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset


class TarIndex:
    """O(1) random access to images inside tar files via byte-offset index.

    Index format (from create_tars.py):
      .idx.json: {"relative/path/frame.jpg": [byte_offset, file_size], ...}

    Images are read by opening the tar as a raw file, seeking to the
    data offset, reading size bytes, and decoding with cv2.imdecode().
    This bypasses tarfile entirely for true O(1) seek performance.

    On first construction the merged index is cached to tar_index.pkl
    in tar_dir. Subsequent loads (including DataLoader worker spawns)
    read from the pickle in ~5 s instead of ~5 min.
    """

    def __init__(self, tar_dir):
        self.tar_dir = Path(tar_dir)
        self.index = {}  # relpath -> (tar_path, data_offset, file_size)
        self._open_handles = {}  # tar_path -> file handle (lazy, LRU'd)

        cache_path = self.tar_dir / "tar_index.pkl"

        if cache_path.exists():
            logger.info(f"Loading tar index from cache {cache_path}...")
            with open(cache_path, "rb") as f:
                self.index = pickle.load(f)
            logger.info(f"TarIndex: {len(self.index):,} files (from cache)")
            return

        idx_files = sorted(self.tar_dir.glob("*.idx.json"))
        if not idx_files:
            raise FileNotFoundError(f"No .idx.json files found in {tar_dir}")

        logger.info(f"Loading {len(idx_files)} tar indexes from {tar_dir}...")
        for idx_file in idx_files:
            tar_name = idx_file.stem.replace(".idx", "")
            tar_path = self.tar_dir / f"{tar_name}.tar"
            if not tar_path.exists():
                logger.warning(f"Tar file not found: {tar_path}, skipping index {idx_file.name}")
                continue
            with open(idx_file) as f:
                seq_index = json.load(f)
            for relpath, (offset, size) in seq_index.items():
                self.index[relpath] = (str(tar_path), offset, size)

        logger.info(f"TarIndex: {len(self.index):,} files across {len(idx_files)} tars")
        logger.info(f"Saving tar index cache to {cache_path}...")
        with open(cache_path, "wb") as f:
            pickle.dump(self.index, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Cache saved.")

    def read_image(self, file_name):
        """Read and decode an image from its indexed tar file.

        Args:
            file_name: relative path as stored in COCO annotations,
                       e.g. '0001/rgb_frame/frame_000001.png'

        Returns:
            numpy array (BGR) or None if not found
        """
        entry = self.index.get(file_name)
        if entry is None:
            return None

        tar_path, offset, size = entry

        # Use cached handle or open new one
        if tar_path in self._open_handles:
            fh = self._open_handles[tar_path]
        else:
            fh = open(tar_path, "rb")
            # Simple LRU: keep at most 8 open handles
            if len(self._open_handles) >= 8:
                oldest = next(iter(self._open_handles))
                self._open_handles[oldest].close()
                del self._open_handles[oldest]
            self._open_handles[tar_path] = fh

        fh.seek(offset)
        raw = fh.read(size)
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        return img

    def close(self):
        for fh in self._open_handles.values():
            fh.close()
        self._open_handles.clear()


def remove_useless_info(coco):
    """
    Remove useless info in coco dataset. COCO object is modified inplace.
    This function is mainly used for saving memory (save about 30% mem).
    """
    if isinstance(coco, COCO):
        dataset = coco.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
        if "annotations" in coco.dataset:
            for anno in coco.dataset["annotations"]:
                anno.pop("segmentation", None)


class COCODataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="instances_train2017.json",
        name="",
        img_size=(416, 416),
        preproc=None,
        cache=False,
        use_tar=False,
        tar_dir=None,
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory (with individual image files OR tar shards)
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
            use_tar (bool): read images from tar shards via byte-offset index
            tar_dir (str): directory containing .tar and .idx.json files.
                           If None, defaults to data_dir.
        """
        super().__init__(img_size)
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "COCO")
        self.data_dir = data_dir
        self.json_file = json_file
        self.use_tar = use_tar
        self.tar_index = None

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        remove_useless_info(self.coco)
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in self.cats])
        self.imgs = None
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.annotations = self._load_coco_annotations()

        if use_tar:
            _tar_dir = tar_dir if tar_dir is not None else data_dir
            self.tar_index = TarIndex(_tar_dir)

        if cache and not use_tar:
            self._cache_images()
        elif cache and use_tar:
            logger.warning("Image caching is not supported with use_tar=True. Ignoring cache.")

    def __len__(self):
        return len(self.ids)

    def __del__(self):
        del self.imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 200G+ RAM and 136G available disk space for training COCO.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = os.path.join(self.data_dir, f"img_resized_cache_{self.name}.array")
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the first time. This might take about 20 minutes for COCO"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x),
                range(len(self.annotations)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.annotations))
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!\n"
                "Everytime the self.input_size is changed in your exp file, you need to delete\n"
                "the cached data and re-generate them.\n"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]

        if self.use_tar and self.tar_index is not None:
            # Annotation file paths include a prefix (e.g. "train/0001/rgb_frame/0001.jpg")
            # but tar indexes use the sequence name only (e.g. "0001/rgb_frame/0001.jpg").
            # Try raw path first, then without the first path component.
            img = self.tar_index.read_image(file_name)
            if img is None:
                parts = file_name.split('/', 1)
                if len(parts) > 1:
                    img = self.tar_index.read_image(parts[1])
            if img is not None:
                return img
            logger.warning(f"Image {file_name} not found in tar index, falling back to disk")

        img_file = os.path.join(self.data_dir, self.name, file_name)
        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"

        return img

    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, resized_info, _ = self.annotations[index]
        if self.imgs is not None:
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)

        return img, res.copy(), img_info, np.array([id_])

    @Dataset.resize_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id
