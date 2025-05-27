#############
# Description: Modified from "Connecting Gaze, Scene, and Attention: Generalized Attention Estimation via Joint Modeling of Gaze and Scene Saliency"
# Author: Chong et al.
# URL: https://github.com/ejcgt/attention-target-detection/blob/master/dataset.py#L29
############

from data.dataset import DatasetBase
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms.functional import (
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
)
from torchvision.ops import box_iou
from data.utils.box_ops import box_xyxy_to_cxcywh, get_head_labelmap
from data.utils.gaze_ops import (
    get_label_map,
    get_label_line_map,
)


from transformers import CLIPTokenizer
import math

class Dataset(DatasetBase):
    def __init__(self, args, mode):
        super().__init__(args, mode)

        filename = "test_annotations_release.txt" if mode == 'test' else "train_annotations_release.txt"
        labels_path = os.path.join(args.data_dir, filename)

        column_names = [
            "path",
            "idx",
            "body_bbox_x",
            "body_bbox_y",
            "body_bbox_w",
            "body_bbox_h",
            "eye_x",
            "eye_y",
            "gaze_x",
            "gaze_y",
            "bbox_x_min",
            "bbox_y_min",
            "bbox_x_max",
            "bbox_y_max",
        ]

        if mode != 'test':
            column_names.append("inout")
        column_names.append("meta")

        df = pd.read_csv(labels_path, sep=",", names=column_names, usecols=column_names, index_col=False)
        
        if mode != 'test':
            df = df.sample(frac=args.down_fact_train, random_state=42)
        else:
            df = df.sample(frac=args.down_fact_test, random_state=42)
            
        # Only use "in" or "out "gaze (-1 is invalid, 0 is out gaze)
        if mode != 'test':
            df = df[df["inout"] != -1]

        # drop invalid head box
        coords = torch.tensor(
            np.array(
                (
                    df["bbox_x_min"].values,
                    df["bbox_y_min"].values,
                    df["bbox_x_max"].values,
                    df["bbox_y_max"].values,
                )
            ).transpose(1, 0)
        )
        valid_bboxes = (coords[:, 2:] > coords[:, :2]).all(dim=1)

        df = df.loc[valid_bboxes.tolist(), :]
        df.reset_index(inplace=True)
        
            
        # -----------
        # Validation data...
        # ----------
        if args.no_validation == 0:
            val_list = np.load(f'{args.data_dir}/val_list_gazefollow.npy')
            if mode == 'val':
                df = df.loc[df['path'].isin(val_list)]
                df.reset_index(inplace=True)
            else:
                df = df.loc[~df['path'].isin(val_list)]
                df.reset_index(inplace=True)
            
        df = df.groupby("path")
        self.keys_gazefollow = sorted(list(df.groups.keys()))
        self.X_gazefollow = df
        
        self.data_dir = args.data_dir
        self.mode = mode
        self.transforms = self.get_transform()
        self.length = len(self.keys_gazefollow)
        
        self.faces_bbox_overflow_coeff = 0.1
        self.gaze_heatmap_default_value = args.gaze_heatmap_default_value
        self.gaze_heatmap_size = args.gaze_heatmap_size
        self.num_queries = args.num_queries

        self.head_heatmap_sigma = args.head_heatmap_sigma
        self.additional_connect = args.additional_connect
        
        self.use_pseudo_head = args.use_pseudo_head
        
        if args.use_pseudo_head != 0:
            self._prepare_aux_faces_dataset()


    def __getitem__(self, index: int):
        if self.mode != 'test':
            return self.__get_train_item__(index)
        else:
            return self.__get_test_item__(index)
      
    def _prepare_aux_faces_dataset(self):
        postfix = '_yolov5'
        labels_path = os.path.join(self.data_dir,
            f'train_heads{postfix}.csv' if self.mode != 'test' else f'test_heads{postfix}.csv',
        )

        column_names = [
            "path",
            "score",
            "head_bbox_x_min",
            "head_bbox_y_min",
            "head_bbox_x_max",
            "head_bbox_y_max",
        ]

        df = pd.read_csv(
            labels_path,
            sep=",",
            names=column_names,
            usecols=column_names,
            skiprows=[
                0,
            ],
            index_col=False,
        )

        # Keep only heads with high score
        df = df[df["score"] >= 0.9]

        # Drop rows with invalid bboxes
        coords = torch.tensor(
            np.array(
                (
                    df["head_bbox_x_min"].values,
                    df["head_bbox_y_min"].values,
                    df["head_bbox_x_max"].values,
                    df["head_bbox_y_max"].values,
                )
            ).transpose(1, 0)
        )
        valid_bboxes = (coords[:, 2:] >= coords[:, :2]).all(dim=1)

        df = df.loc[valid_bboxes.tolist(), :]
        df.reset_index(inplace=True)
        df = df.groupby("path")

        self.X_faces_aux = df
        self.keys_faces_aux = list(df.groups.keys())
          
    def __get_train_item__(self, index: int):
        # Load image
        img = Image.open(os.path.join(self.data_dir, self.keys_gazefollow[index]))
        img = img.convert("RGB")
        img_width, img_height = img.size
        
        boxes = []
        gaze_points = []
        gaze_heatmaps = []
        head_heatmaps = []
        gaze_watch_outside = []
        for _, row in self.X_gazefollow.get_group(self.keys_gazefollow[index]).iterrows():
            box_x_min = row["bbox_x_min"]
            box_y_min = row["bbox_y_min"]
            box_x_max = row["bbox_x_max"]
            box_y_max = row["bbox_y_max"]

            # Expand bbox
            box_width = box_x_max - box_x_min
            box_height = box_y_max - box_y_min
            box_x_min -= box_width * self.faces_bbox_overflow_coeff
            box_y_min -= box_height * self.faces_bbox_overflow_coeff
            box_x_max += box_width * self.faces_bbox_overflow_coeff
            box_y_max += box_height * self.faces_bbox_overflow_coeff

            # Jitter
            if np.random.random_sample() <= 0.5:
                bbox_overflow_coeff = np.random.random_sample() * 0.2
                box_x_min -= box_width * bbox_overflow_coeff
                box_y_min -= box_height * bbox_overflow_coeff
                box_x_max += box_width * bbox_overflow_coeff
                box_y_max += box_height * bbox_overflow_coeff

            boxes.append(
                torch.FloatTensor([box_x_min, box_y_min, box_x_max, box_y_max])
            )

            # Gaze point
            gaze_x = row["gaze_x"] * img_width
            gaze_y = row["gaze_y"] * img_height
            gaze_points.append(torch.FloatTensor([gaze_x, gaze_y]).view(1, 2))

            # Gaze watch outside
            gaze_watch_outside.append(row["inout"] == 0)

        aux_faces_boxes_out = [] 
        if (
            self.use_pseudo_head != 0
            and self.keys_gazefollow[index] in self.keys_faces_aux
        ):
            aux_faces_boxes = []
            for _, row in self.X_faces_aux.get_group(
                self.keys_gazefollow[index]
            ).iterrows():
                # Face bbox
                box_x_min = row["head_bbox_x_min"]
                box_y_min = row["head_bbox_y_min"]
                box_x_max = row["head_bbox_x_max"]
                box_y_max = row["head_bbox_y_max"]

                box_width = box_x_max - box_x_min
                box_height = box_y_max - box_y_min
                box_x_min -= box_width * self.faces_bbox_overflow_coeff
                box_y_min -= box_height * self.faces_bbox_overflow_coeff
                box_x_max += box_width * self.faces_bbox_overflow_coeff
                box_y_max += box_height * self.faces_bbox_overflow_coeff

                aux_faces_boxes.append(
                    torch.FloatTensor([box_x_min, box_y_min, box_x_max, box_y_max])
                )
            
            # Calculate iou between boxes and aux_head_boxes and remove from aux_face_boxes
            # the boxes where iou is not zero
            iou = box_iou(
                torch.stack(boxes),
                torch.stack(aux_faces_boxes),
            ) 
            aux_faces_boxes_out = [
                aux_faces_boxes[i]
                for i in range(len(aux_faces_boxes))
                if iou[:, i].max() == 0
            ]
            
        for i in range(len(boxes)):
            aux_faces_boxes_out.append(boxes[i])
        
        # Random color change
        if np.random.random_sample() <= 0.5:
            img = adjust_brightness(img, brightness_factor=np.random.uniform(0.5, 1.5))
            img = adjust_contrast(img, contrast_factor=np.random.uniform(0.5, 1.5))
            img = adjust_saturation(img, saturation_factor=np.random.uniform(0, 1.5))

        target = {
            "path": self.keys_gazefollow[index],
            "img_size": torch.FloatTensor([img_height, img_width]),
            "boxes": torch.stack(boxes),
            "gaze_points": torch.stack(gaze_points),
            "gaze_watch_outside": torch.BoolTensor(gaze_watch_outside).long(),
        }
        
        if self.use_pseudo_head != 0:
            target['aux_faces_boxes'] = torch.stack(aux_faces_boxes_out) # H, 4

        # Transform image and rescale all bounding target
        img, target = self.transforms(img, target)
        img_height, img_width = target["img_size"]

        num_boxes = len(target["boxes"])

        boxes = torch.full((num_boxes, 4), 0).float()
        boxes[: len(target["boxes"]), :] = target["boxes"]
        target["boxes"] = boxes
         
        img_size = target["img_size"].repeat(num_boxes, 1)
        target["img_size"] = img_size

        gaze_points = torch.full((num_boxes, 1, 2), -1).float()
        gaze_points[: len(target["gaze_points"]), :, :] = target["gaze_points"]
        target["gaze_points"] = gaze_points

        gaze_watch_outside = torch.full((num_boxes, 1), 1).float()
        gaze_watch_outside[: len(target["gaze_watch_outside"]), 0] = target[ "gaze_watch_outside"]
        target["gaze_watch_outside"] = gaze_watch_outside.long()

        for gaze_point, gaze_watch_outside in zip(
            target["gaze_points"], target["gaze_watch_outside"]
        ):
            gaze_x, gaze_y = gaze_point.squeeze(0) # H x 1 x 2
            if gaze_watch_outside == 0:
                sigma = 3
                sigma = int(sigma * self.gaze_heatmap_size / 64)
                gaze_heatmap = get_label_map(
                    torch.zeros((self.gaze_heatmap_size, self.gaze_heatmap_size)),
                    [
                        gaze_x * self.gaze_heatmap_size,
                        gaze_y * self.gaze_heatmap_size,
                    ],
                    sigma,
                    pdf="Gaussian",
                )
            else:
                gaze_heatmap = torch.full(
                    (self.gaze_heatmap_size, self.gaze_heatmap_size),
                    float(self.gaze_heatmap_default_value),
                )

            gaze_heatmaps.append(gaze_heatmap)

        target["gaze_heatmaps"] = torch.stack(gaze_heatmaps)

        # [head heatmap] loop through each head
        for (bbox_cx, bbox_cy, bbox_w, bbox_h) in target["boxes"]:
            sigma_x = bbox_w / self.head_heatmap_sigma
            sigma_y = bbox_h / self.head_heatmap_sigma
            
            head_heatmap = get_head_labelmap(
                    torch.zeros(
                        (self.gaze_heatmap_size, self.gaze_heatmap_size)
                    ),
                    [
                        bbox_cx * self.gaze_heatmap_size,
                        bbox_cy * self.gaze_heatmap_size,
                    ],
                    [
                        sigma_x * self.gaze_heatmap_size,
                        sigma_y * self.gaze_heatmap_size,
                    ],
                    pdf="Gaussian",
                )

            head_heatmaps.append(head_heatmap)

        target["head_heatmaps"] = torch.stack(head_heatmaps)
        
        if self.use_pseudo_head != 0:
            head_heatmaps_aux = []
            for bbox in target["aux_faces_boxes"]:
                # cxcywh
                bbox_cx, bbox_cy, bbox_w, bbox_h = bbox

                sigma_x = bbox_w / self.head_heatmap_sigma
                sigma_y = bbox_h / self.head_heatmap_sigma
                
                head_heatmap = get_head_labelmap(
                        torch.zeros(
                            (self.gaze_heatmap_size, self.gaze_heatmap_size)
                        ),
                        [
                            bbox_cx * self.gaze_heatmap_size,
                            bbox_cy * self.gaze_heatmap_size,
                        ],
                        [
                            sigma_x * self.gaze_heatmap_size,
                            sigma_y * self.gaze_heatmap_size,
                        ],
                        pdf="Gaussian",
                    )

                head_heatmaps_aux.append(head_heatmap)

            target["head_heatmaps_all"], _ = torch.max(torch.stack(head_heatmaps_aux), dim=0) 
        else:
            target["head_heatmaps_all"], _ = torch.max(target["head_heatmaps"], dim=0) 
        
        if self.additional_connect != 0:
            # first get line map
            connect_heatmaps = []
            # [head heatmap] loop through each head
            for (bbox_cx, bbox_cy, bbox_w, bbox_h), \
                gaze_point, gaze_watch_outside, \
                gaze_heatmap, head_heatmap in zip(
                target["boxes"],
                target["gaze_points"], target["gaze_watch_outside"],
                target["gaze_heatmaps"], target["head_heatmaps"]
            ):
                #if not regression_is_padding and gaze_watch_outside == 0:
                if gaze_watch_outside == 0:
                    
                    gaze_x, gaze_y = gaze_point.squeeze(0) # H x 1 x 2
                    x1, y1 = bbox_cx * self.gaze_heatmap_size, bbox_cy * self.gaze_heatmap_size
                    x2, y2 = gaze_x * self.gaze_heatmap_size, gaze_y * self.gaze_heatmap_size
                    line_x = np.linspace(x1, x2)
                    line_y = np.linspace(y1, y2)
                    points = torch.from_numpy(np.concatenate((line_x[..., None], line_y[..., None]), axis=1))
                        
                    sigma_h_x = bbox_w / self.head_heatmap_sigma  * self.gaze_heatmap_size
                    sigma_h_y = bbox_h / self.head_heatmap_sigma  * self.gaze_heatmap_size
                    sigma_g = int(3 * self.gaze_heatmap_size / 64)
                    
                    sigma_x = np.linspace(sigma_h_x, sigma_g, points.shape[0]) 
                    sigma_y = np.linspace(sigma_h_y, sigma_g, points.shape[0])
                    sigmas = torch.from_numpy(np.concatenate((sigma_x[..., None], sigma_y[..., None]), axis=1))
                    
                    scalars = np.linspace(1, 1, points.shape[0]) 
                    
                    line_heatmap = get_label_line_map(
                        torch.zeros((self.gaze_heatmap_size, self.gaze_heatmap_size)),
                        points,
                        sigmas,
                        scalars,
                        pdf='Gaussian'
                    )
                else:
                    line_heatmap = torch.full(
                        (self.gaze_heatmap_size, self.gaze_heatmap_size),
                        float(self.gaze_heatmap_default_value),
                    )
                connect_heatmaps.append(line_heatmap)
            
            target["connect_heatmaps"] = torch.stack(connect_heatmaps)
               
        return img, target
    
    def __get_test_item__(self, index: int):
        # Load image
        img = Image.open(os.path.join(self.data_dir, self.keys_gazefollow[index]))
        img = img.convert("RGB")
        img_width, img_height = img.size

        boxes = []
        gaze_points = []
        gaze_points_is_padding = []
        gaze_heatmaps = []
        head_heatmaps = []
        gaze_watch_outside = []

        # Group annotations from same scene with same person
        for _, same_person_annotations in self.X_gazefollow.get_group(self.keys_gazefollow[index]).groupby(['bbox_x_min', 'bbox_y_min']):
            # Group annotations of the same person
            sp_gaze_points = []
            sp_boxes = []
            sp_gaze_inside = []
            for _, row in same_person_annotations.iterrows():
                # Load bbox
                box_x_min = row["bbox_x_min"]
                box_y_min = row["bbox_y_min"]
                box_x_max = row["bbox_x_max"]
                box_y_max = row["bbox_y_max"]

                sp_boxes.append(
                    torch.FloatTensor([box_x_min, box_y_min, box_x_max, box_y_max])
                )

                gaze_x = row["gaze_x"] * img_width
                gaze_y = row["gaze_y"] * img_height
                sp_gaze_points.append(torch.FloatTensor([gaze_x, gaze_y]))
                sp_gaze_inside.append(True)

            boxes.append(torch.FloatTensor(sp_boxes[-1]))

            sp_gaze_points_padded = torch.full((20, 2), -1).float()
            sp_gaze_points_padded[: len(sp_gaze_points), :] = torch.stack(
                sp_gaze_points
            )
            sp_gaze_points_is_padding = torch.full((20,), False)
            sp_gaze_points_is_padding[len(sp_gaze_points) :] = True

            gaze_points.append(sp_gaze_points_padded)
            gaze_points_is_padding.append(sp_gaze_points_is_padding)

            gaze_watch_outside.append(
                (
                    torch.BoolTensor(sp_gaze_inside).sum() < len(sp_gaze_inside) / 2
                ).item()
            )
        
        
        aux_faces_boxes_out = []
        if (
            self.use_pseudo_head != 0
            and self.keys_gazefollow[index] in self.keys_faces_aux
        ): 
            aux_faces_boxes = []
            for _, row in self.X_faces_aux.get_group(
                self.keys_gazefollow[index]
            ).iterrows():
                # Face bbox
                box_x_min = row["head_bbox_x_min"]
                box_y_min = row["head_bbox_y_min"]
                box_x_max = row["head_bbox_x_max"]
                box_y_max = row["head_bbox_y_max"]

                aux_faces_boxes.append(
                    torch.FloatTensor([box_x_min, box_y_min, box_x_max, box_y_max])
                )
                
            
            # Calculate iou between boxes and aux_head_boxes and remove from aux_face_boxes
            # the boxes where iou is not zero
            iou = box_iou(
                torch.stack(boxes),
                torch.stack(aux_faces_boxes),
            )
            aux_faces_boxes_out = [
                aux_faces_boxes[i]
                for i in range(len(aux_faces_boxes))
                if iou[:, i].max() == 0
            ]
            
        for i in range(len(boxes)):
            aux_faces_boxes_out.append(boxes[i])
            

        target = {
            "path": self.keys_gazefollow[index],
            "img_size": torch.FloatTensor([img_height, img_width]),
            "boxes": torch.stack(boxes),
            "gaze_points": torch.stack(gaze_points),
            "gaze_points_is_padding": torch.stack(gaze_points_is_padding),
            "gaze_watch_outside": torch.BoolTensor(gaze_watch_outside).long(),
        }
        
        if self.use_pseudo_head != 0:
            target['aux_faces_boxes'] = torch.stack(aux_faces_boxes_out) # H, 4

        # Transform image and rescale all bounding target
        img, target = self.transforms(
            img,
            target,
        )
        img_height, img_width = target["img_size"]

        num_boxes = len(target["boxes"])
        
        img_size = target["img_size"].repeat(num_boxes, 1)
        target["img_size"] = img_size

        boxes = torch.full((num_boxes, 4), 0).float()
        boxes[: len(target["boxes"]), :] = target["boxes"]
        boxes[len(target["boxes"]) :, :] = box_xyxy_to_cxcywh(torch.tensor([0, 0, 1, 1]))
        target["boxes"] = boxes
        
        gaze_points = torch.full((num_boxes, 20, 2), -1).float()
        gaze_points[: len(target["gaze_points"]), :, :] = target["gaze_points"]
        target["gaze_points"] = gaze_points

        gaze_points_is_padding = torch.full((num_boxes, 20), True)
        gaze_points_is_padding[: len(target["gaze_points_is_padding"]), :] = target[
            "gaze_points_is_padding"
        ]
        target["gaze_points_is_padding"] = gaze_points_is_padding

        gaze_watch_outside = torch.full((num_boxes, 1), 1).float()
        gaze_watch_outside[: len(target["gaze_watch_outside"]), 0] = target[
            "gaze_watch_outside"
        ]
        target["gaze_watch_outside"] = gaze_watch_outside.long()
        
        for gaze_points, gaze_point_is_padding, gaze_watch_outside in zip(
            target["gaze_points"],
            target["gaze_points_is_padding"],
            target["gaze_watch_outside"]
        ):
            if gaze_watch_outside == 0:
                gaze_heatmap = []

                for (gaze_x, gaze_y), gaze_is_padding in zip(
                    gaze_points, gaze_point_is_padding
                ):
                    if gaze_x == -1 or gaze_y == -1 or gaze_is_padding:
                        continue
                    
                    sigma = 3
                    sigma = int(sigma * self.gaze_heatmap_size / 64)
                    
                    gaze_heatmap.append(
                        get_label_map(
                            torch.zeros(
                                (self.gaze_heatmap_size, self.gaze_heatmap_size)
                            ),
                            [
                                gaze_x * self.gaze_heatmap_size,
                                gaze_y * self.gaze_heatmap_size,
                            ],
                            sigma,
                            pdf="Gaussian",
                        )
                    )

                gaze_heatmap = torch.stack(gaze_heatmap)
                gaze_heatmap = gaze_heatmap.sum(dim=0) / gaze_heatmap.sum(dim=0).max()
            else:
                gaze_heatmap = torch.full(
                    (self.gaze_heatmap_size, self.gaze_heatmap_size),
                    self.gaze_heatmap_default_value,
                )

            gaze_heatmaps.append(gaze_heatmap)

        target["gaze_heatmaps"] = torch.stack(gaze_heatmaps)

        # [head heatmap] loop through each head
        for bbox_cx, bbox_cy, bbox_w, bbox_h in target["boxes"]:
            head_heatmap = []
            
            sigma_x = bbox_w / self.head_heatmap_sigma
            sigma_y = bbox_h / self.head_heatmap_sigma
            
            head_heatmap = get_head_labelmap(
                    torch.zeros(
                        (self.gaze_heatmap_size, self.gaze_heatmap_size)
                    ),
                    [
                        bbox_cx * self.gaze_heatmap_size,
                        bbox_cy * self.gaze_heatmap_size,
                    ],
                    [
                        sigma_x * self.gaze_heatmap_size,
                        sigma_y * self.gaze_heatmap_size,
                    ],
                    pdf="Gaussian",
                )

            head_heatmaps.append(head_heatmap)

        target["head_heatmaps"] = torch.stack(head_heatmaps) # H x 4
        
        if self.use_pseudo_head != 0:
            head_heatmaps_aux = []
            for bbox_cx, bbox_cy, bbox_w, bbox_h in target["aux_faces_boxes"]:
                
                sigma_x = bbox_w / self.head_heatmap_sigma
                sigma_y = bbox_h / self.head_heatmap_sigma
                
                head_heatmap = get_head_labelmap(
                        torch.zeros(
                            (self.gaze_heatmap_size, self.gaze_heatmap_size)
                        ),
                        [
                            bbox_cx * self.gaze_heatmap_size,
                            bbox_cy * self.gaze_heatmap_size,
                        ],
                        [
                            sigma_x * self.gaze_heatmap_size,
                            sigma_y * self.gaze_heatmap_size,
                        ],
                        pdf="Gaussian",
                    )

                head_heatmaps_aux.append(head_heatmap)

            target["head_heatmaps_all"], _ = torch.max(torch.stack(head_heatmaps_aux), dim=0) 
        else:
            target["head_heatmaps_all"], _ = torch.max(target["head_heatmaps"], dim=0) 
        
        if self.additional_connect != 0:
            # first get line map
            connect_heatmaps = []
            # [head heatmap] loop through each head
            for (bbox_cx, bbox_cy, bbox_w, bbox_h), \
                gaze_point, gaze_watch_outside, \
                gaze_heatmap, head_heatmap, gaze_points_is_padding in zip(
                target["boxes"],
                target["gaze_points"], target["gaze_watch_outside"],
                target["gaze_heatmaps"], target["head_heatmaps"], target["gaze_points_is_padding"]
            ):  
                if gaze_watch_outside == 0:
                    
                    gaze_x, gaze_y = gaze_point[~gaze_points_is_padding].mean(dim=0) # 20 x 2
                    x1, y1 = bbox_cx * self.gaze_heatmap_size, bbox_cy * self.gaze_heatmap_size
                    x2, y2 = gaze_x * self.gaze_heatmap_size, gaze_y * self.gaze_heatmap_size
                    line_x = np.linspace(x1, x2)
                    line_y = np.linspace(y1, y2)
                    points = torch.from_numpy(np.concatenate((line_x[..., None], line_y[..., None]), axis=1))
                        
                    sigma_h_x = bbox_w / self.head_heatmap_sigma  * self.gaze_heatmap_size
                    sigma_h_y = bbox_h / self.head_heatmap_sigma  * self.gaze_heatmap_size
                    sigma_g = int(3 * self.gaze_heatmap_size / 64)
                    
                    sigma_x = np.linspace(sigma_h_x, sigma_g, points.shape[0]) 
                    sigma_y = np.linspace(sigma_h_y, sigma_g, points.shape[0])
                    sigmas = torch.from_numpy(np.concatenate((sigma_x[..., None], sigma_y[..., None]), axis=1))
                    
                    scalars = np.linspace(1, 1, points.shape[0]) 
                    
                    line_heatmap = get_label_line_map(
                        torch.zeros((self.gaze_heatmap_size, self.gaze_heatmap_size)),
                        points,
                        sigmas,
                        scalars,
                        pdf='Gaussian'
                    )
                else:
                    line_heatmap = torch.full(
                        (self.gaze_heatmap_size, self.gaze_heatmap_size),
                        float(self.gaze_heatmap_default_value),
                    )
                connect_heatmaps.append(line_heatmap)
            
            target["connect_heatmaps"] = torch.stack(connect_heatmaps)
          
        return img, target

    def __len__(self):
        return self.length