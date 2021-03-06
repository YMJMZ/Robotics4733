from typing import Tuple, Optional, Dict
import cv2

import numpy as np
from matplotlib import cm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from torchvision.transforms.functional import rotate
import copy

from common import draw_grasp


def get_gaussian_scoremap(
        shape: Tuple[int, int], 
        keypoint: np.ndarray, 
        sigma: float=1, dtype=np.float32) -> np.ndarray:
    """
    Generate a image of shape=:shape:, generate a Gaussian distribtuion
    centered at :keypont: with standard deviation :sigma: pixels.
    keypoint: shape=(2,)
    """
    coord_img = np.moveaxis(np.indices(shape),0,-1).astype(dtype)
    sqrt_dist_img = np.square(np.linalg.norm(
        coord_img - keypoint[::-1].astype(dtype), axis=-1))
    scoremap = np.exp(-0.5/np.square(sigma)*sqrt_dist_img)
    return scoremap

class AffordanceDataset(Dataset):
    """
    Transformational dataset.
    raw_dataset is of type train.RGBDataset
    """
    def __init__(self, raw_dataset: Dataset):
        super().__init__()
        self.raw_dataset = raw_dataset
    
    def __len__(self) -> int:
        return len(self.raw_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Transform the raw RGB dataset element into
        training targets for AffordanceModel.
        return: 
        {
            'input': torch.Tensor (3,H,W), torch.float32, range [0,1]
            'target': torch.Tensor (1,H,W), torch.float32, range [0,1]
        }
        Note: self.raw_dataset[idx]['rgb'] is torch.Tensor (H,W,3) torch.uint8
        """
        # checkout train.RGBDataset for the content of data
        data = self.raw_dataset[idx]
        # TODO: complete this method
        # Hint: Use get_gaussian_scoremap
        # Hint: https://imgaug.readthedocs.io/en/latest/source/examples_keypoints.html
        # ===============================================================================
        coord = data['center_point'].numpy()
        kps = KeypointsOnImage([
            Keypoint(x=coord[0], y=coord[1])
        ], shape=(128,128))
        rot = iaa.Affine(
            rotate=-1*data['angle'].numpy().item()
        )
        image_aug, kps_aug = rot(
            image=np.concatenate(
                [np.expand_dims(data['scoremap'].numpy(), axis=-1),data['rgb'].numpy()], 
                axis=-1),
            keypoints = kps
        )
        coord_aug = np.array([kps_aug[0].x, kps_aug[0].y])
        # target = torch.from_numpy(
        #     get_gaussian_scoremap(
        #         shape=[data['rgb'].size()[0], data['rgb'].size()[1]],
        #         keypoint=coord_aug
        #     )
        alpha = 1
        scoremap = (alpha*image_aug[:,:,0] + get_gaussian_scoremap(
                shape=[data['rgb'].size()[0], data['rgb'].size()[1]],
                keypoint=coord_aug
            ))
        scoremap /= np.max(scoremap)
        target = torch.from_numpy(scoremap).type(torch.float32)
        result = {
            'input': torch.from_numpy(image_aug[:,:,1:]).type(torch.float32).permute(2, 0, 1)/255,
            'target': torch.unsqueeze(target, 0),
        }
        return result
        # ===============================================================================

class AffordanceModel(nn.Module):
    def __init__(self, n_channels: int=3, n_classes: int=1, **kwargs):
        """
        A simplified U-Net with twice of down/up sampling and single convolution.
        ref: https://arxiv.org/abs/1505.04597, https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        :param n_channels (int): number of channels (for grayscale 1, for rgb 3)
        :param n_classes (int): number of segmentation classes (num objects + 1 for background)
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = nn.Sequential(
            # For simplicity:
            #     use padding 1 for 3*3 conv to keep the same Width and Height and ease concatenation
            nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential( 
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.outc = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)
        # hack to get model device
        self.dummy_param = nn.Parameter(torch.empty(0))

    @property
    def device(self) -> torch.device:
        return self.dummy_param.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_inc = self.inc(x)  # N * 64 * H * W
        x_down1 = self.down1(x_inc)  # N * 128 * H/2 * W/2
        x_down2 = self.down2(x_down1)  # N * 256 * H/4 * W/4
        x_up1 = self.upconv1(x_down2)  # N * 128 * H/2 * W/2
        x_up1 = torch.cat([x_up1, x_down1], dim=1)  # N * 256 * H/2 * W/2
        x_up1 = self.conv1(x_up1)  # N * 128 * H/2 * W/2
        x_up2 = self.upconv2(x_up1)  # N * 64 * H * W
        x_up2 = torch.cat([x_up2, x_inc], dim=1)  # N * 128 * H * W
        x_up2 = self.conv2(x_up2)  # N * 64 * H * W
        x_outc = self.outc(x_up2)  # N * n_classes * H * W
        return x_outc

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict affordance using this model.
        This is required due to BCEWithLogitsLoss
        """
        return torch.sigmoid(self.forward(x))

    @staticmethod
    def get_criterion() -> torch.nn.Module:
        """
        Return the Loss object needed for training.
        Hint: why does nn.BCELoss does not work well?
        """
        return nn.BCEWithLogitsLoss()

    @staticmethod
    def visualize(input: np.ndarray, output: np.ndarray, 
            target: Optional[np.ndarray]=None) -> np.ndarray:
        """
        Visualize rgb input and affordance as a single rgb image.
        """
        cmap = cm.get_cmap('viridis')
        in_img = np.moveaxis(input, 0, -1)
        pred_img = cmap(output[0])[...,:3]
        row = [in_img, pred_img]
        if target is not None:
            gt_img = cmap(target[0])[...,:3]
            row.append(gt_img)
        img = (np.concatenate(row, axis=1)*255).astype(np.uint8)
        return img


    def predict_grasp(self, rgb_obs: np.ndarray
            ) -> Tuple[Tuple[int, int], float, np.ndarray]:
        """
        Given a RGB image observation, predict the grasping location and angle in image space.
        return coord, angle, vis_img
        :coord: tuple(int x, int y). By OpenCV convension, x is left-to-right and y is top-to-bottom.
        :angle: float. By OpenCV convension, angle is clockwise rotation.
        :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.

        Note: torchvision's rotation is counter clockwise, while imgaug,OpenCV's rotation are clockwise.
        """
        device = self.device
        # TODO: complete this method (prediction)
        # Hint: why do we provide the model's device here?
        # ===============================================================================
        coord, angle = None, None
        rgb_tensor = torch.from_numpy(rgb_obs).permute(2, 0, 1)
        max_predict = 0.0
        best_scoremap = None
        best_input = None
        scores = []
        visual_map = []
        temp = []
        index = None
        for i in range(8):
            rotated_rgb = rotate(rgb_tensor, 22.5*i)
            rotated_rgb = torch.unsqueeze(rotated_rgb/255, 0)
            rotated_rgb.to(device)
            predict = self.predict(rotated_rgb)
            scores.append(torch.max(predict).item())
            if torch.max(predict) > max_predict:
                location = torch.argmax(predict).item()
                coord = (location%rgb_obs.shape[1], location//rgb_obs.shape[1])
                angle = i*22.5
                best_scoremap = predict.detach().numpy()
                best_input = rotated_rgb.detach().numpy()
                max_predict = torch.max(predict).item()
                index = i
            
            temp_img = self.visualize(
                input=rotated_rgb.detach().numpy()[0,:,:,:],
                output=predict.detach().numpy()[0,:,:,:]
            )
            temp.append(temp_img)
            if i%2==1:
                visual_map.append(temp)
                temp = []

        if angle is None:
            print('What the hell!')
            coord = [0,0]
            angle = 0.0
        
        rot = iaa.Affine(
            rotate=angle
        )
        kps = KeypointsOnImage([
            Keypoint(x=coord[0], y=coord[1])
        ], shape=rgb_obs.shape)
        aug_input, aug_kps = rot(
            image=np.moveaxis(best_scoremap[0,:,:,:], 0, -1), 
            keypoints=kps)
        ori_coord = coord
        coord = (int(aug_kps[0].x), int(aug_kps[0].y))
        # angle = -angle
        print(f'best angle:{angle}')
        print(scores)
            
        # TODO: complete this method (visualization)
        # :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.
        # Hint: use common.draw_grasp
        # Hint: see self.visualize
        # Hint: draw a grey (127,127,127) line on the bottom row of each image.
        # ===============================================================================
        # draw_grasp(rgb_obs, coord, angle)
        # vis_img = self.visualize(
        #     input=best_input[0,:,:,:], 
        #     output=best_scoremap[0,:,:,:],
        #     target=np.expand_dims(get_gaussian_scoremap((128,128),np.array(ori_coord)), axis=0)
        #     )
        draw_grasp(visual_map[index//2][index%2], ori_coord, 0)
        vis_img = cv2.vconcat([cv2.hconcat(l) for l in visual_map])
        # vis_img = rgb_obs
        # ===============================================================================
        return coord, angle, vis_img

