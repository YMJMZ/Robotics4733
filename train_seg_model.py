import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
import image
import numpy as np
from random import seed
from sim import get_tableau_palette
import matplotlib.pyplot as plt


# ==================================================
mean_rgb = [0.485, 0.456, 0.406]
std_rgb = [0.229, 0.224, 0.225]
# ==================================================

class RGBDataset(Dataset):
    def __init__(self, img_dir):
        """
            Initialize instance variables.
            :param img_dir (str): path of train or test folder.
            :return None:
        """
        # TODO: complete this method
        # ===============================================================================
        mean_rgb = [0.722, 0.751, 0.807]
        std_rgb = [0.171, 0.179, 0.197]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=mean_rgb,
            #     std=std_rgb),
        ])

        self.dataset_dir = img_dir
        self.rgb_files = [file for file in os.listdir(os.path.join(img_dir, 'rgb')) 
                          if os.path.isfile(os.path.join(img_dir, 'rgb', file))]
        self.gt_files = [file for file in os.listdir(os.path.join(img_dir, 'gt')) 
                          if os.path.isfile(os.path.join(img_dir, 'gt', file))]
        assert len(self.rgb_files) == len(self.gt_files),\
             "The number of rgb does not match the number of gt."
        self.dataset_length = len(self.rgb_files)
        # ===============================================================================

    def __len__(self):
        """
            Return the length of the dataset.
            :return dataset_length (int): length of the dataset, i.e. number of samples in the dataset
        """
        # TODO: complete this method
        # ===============================================================================
        return self.dataset_length
        # ===============================================================================

    def __getitem__(self, idx):
        """
            Given an index, return paired rgb image and ground truth mask as a sample.
            :param idx (int): index of each sample, in range(0, dataset_length)
            :return sample: a dictionary that stores paired rgb image and corresponding ground truth mask.
        """
        # TODO: complete this method
        # Hint:
        # - Use image.read_rgb() and image.read_mask() to read the images.
        # - Think about how to associate idx with the file name of images.
        # - Remember to apply transform on the sample.
        # ===============================================================================
        rgb_img = image.read_rgb(os.path.join(self.dataset_dir, 'rgb', self.rgb_files[idx]))
        gt_mask = image.read_mask(os.path.join(self.dataset_dir, 'gt', self.gt_files[idx]))
        sample = {'input': self.transform(rgb_img), 'target': torch.LongTensor(gt_mask)}
        return sample
        # ===============================================================================


class miniUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        """
        A simplified U-Net with twice of down/up sampling and single convolution.
        ref: https://arxiv.org/abs/1505.04597, https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        :param n_channels (int): number of channels (for grayscale 1, for rgb 3)
        :param n_classes (int): number of segmentation classes (num objects + 1 for background)
        """
        super(miniUNet, self).__init__()
        # TODO: complete this method
        # ===============================================================================
        self.layer = [n_channels, 16, 32, 64, 128, 256]
        self.single_conv_downs = nn.ModuleList(
            [self.single_conv(in_c, out_c) for in_c, out_c in zip(self.layer[:-1], self.layer[1:])]
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.single_conv_ups = nn.ModuleList(
            [self.single_conv(in_c + out_c, out_c) for in_c, out_c in zip(self.layer[::-1][:-2], self.layer[::-1][1:-1])]
        )
        self.pool = nn.MaxPool2d(2)
        self.out = nn.Conv2d(self.layer[1], n_classes, kernel_size=1)
        # ===============================================================================

    def single_conv(self, in_channel, out_channel):
        conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        return conv

    def forward(self, x):
        # TODO: complete this method
        # ===============================================================================
        # Down sampling
        concat_layers = []
        for down in self.single_conv_downs:
            x = down(x)
            if down != self.single_conv_downs[-1]:
                concat_layers.append(x)
                x = self.pool(x)
        concat_layers = concat_layers[::-1]

        # Up sampling
        for up, concat_layer in zip(self.single_conv_ups, concat_layers):
            x = self.up(x)
            assert (x.shape[2] == concat_layer.shape[2] 
                 or x.shape[3] == concat_layer.shape[3]), \
                    f'concat_layer:{concat_layer.shape[2]}x{concat_layer.shape[3]}' \
                    f'cannot concatenate with input:{x.shape[2]}x{x.shape[3]}'
            x = torch.cat([concat_layer, x], dim=1)
            x = up(x)

        # Output layer
        output = self.out(x)

        return output
        # ===============================================================================


def save_chkpt(model, epoch, test_miou, chkpt_path):
    """
        Save the trained model.
        :param model (torch.nn.module object): miniUNet object in this homework, trained model.
        :param epoch (int): current epoch number.
        :param test_miou (float): miou of the test set.
        :return: None
    """
    state = {'model_state_dict': model.state_dict(),
             'epoch': epoch,
             'model_miou': test_miou, }
    torch.save(state, chkpt_path)
    print("checkpoint saved at epoch", epoch)


def load_chkpt(model, chkpt_path, device):
    """
        Load model parameters from saved checkpoint.
        :param model (torch.nn.module object): miniUNet model to accept the saved parameters.
        :param chkpt_path (str): path of the checkpoint to be loaded.
        :return model (torch.nn.module object): miniUNet model with its parameters loaded from the checkpoint.
        :return epoch (int): epoch at which the checkpoint is saved.
        :return model_miou (float): miou of the test set at the checkpoint.
    """
    checkpoint = torch.load(chkpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    model_miou = checkpoint['model_miou']
    print("epoch, model_miou:", epoch, model_miou)
    return model, epoch, model_miou


def save_prediction(model, dataloader, dump_dir, device, BATCH_SIZE):
    """
        For all datapoints d in dataloader, save  ground truth segmentation mask (as {id}.png)
          and predicted segmentation mask (as {id}_pred.png) in dump_dir.
        :param model (torch.nn.module object): trained miniUNet model
        :param dataloader (torch.utils.data.DataLoader object): dataloader to use for getting predictions
        :param dump_dir (str): dir path for dumping predictions
        :param device (torch.device object): pytorch cpu/gpu device object
        :param BATCH_SIZE (int): batch size of dataloader
        :return: None
    """
    print(f"Saving predictions in directory {dump_dir}")
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    model.eval()
    with torch.no_grad():
        for batch_ID, sample_batched in enumerate(dataloader):
            data, target = sample_batched['input'].to(device), sample_batched['target'].to(device)
            output = model(data)
            _, pred = torch.max(output, dim=1)
            for i in range(pred.shape[0]):
                gt_image = convert_seg_split_into_color_image(target[i].cpu().numpy())
                pred_image = convert_seg_split_into_color_image(pred[i].cpu().numpy())
                combined_image = np.concatenate((gt_image, pred_image), axis=1)
                test_ID = batch_ID * BATCH_SIZE + i
                image.write_mask(combined_image, f"{dump_dir}/{test_ID}_gt_pred.png")


def iou(pred, target, n_classes=6):
    """
        Compute IoU on each object class and return as a list.
        :param pred (np.array object): predicted mask
        :param target (np.array object): ground truth mask
        :param n_classes (int): number of classes
        :return cls_ious (list()): a list of IoU on each object class
    """
    cls_ious = []
    # Flatten
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(1, n_classes):  # class 0 is background
        pred_P = pred == cls
        target_P = target == cls
        pred_N = ~pred_P
        target_N = ~target_P
        if target_P.sum() == 0:
            # print("class", cls, "doesn't exist in target")
            continue
        else:
            intersection = pred_P[target_P].sum()  # TP
            if intersection == 0:
                # print("pred and target for class", cls, "have no intersection")
                continue
            else:
                FP = pred_P[target_N].sum()
                FN = pred_N[target_P].sum()
                union = intersection + FN + FP  # or pred_P.sum() + target_P.sum() - intersection
                cls_ious.append(float(intersection) / float(union))
    return cls_ious


def run(model, loader, criterion, device, is_train=False, optimizer=None):
    """
        Run forward pass for each sample in the dataloader. Run backward pass and optimize if training.
        Calculate and return mean_epoch_loss and mean_iou
        :param model (torch.nn.module object): miniUNet model object
        :param loader (torch.utils.data.DataLoader object): dataloader 
        :param criterion (torch.nn.module object): Pytorch criterion object
        :param is_train (bool): True if training
        :param optimizer (torch.optim.Optimizer object): Pytorch optimizer object
        :return mean_epoch_loss (float): mean loss across this epoch
        :return mean_iou (float): mean iou across this epoch
    """
    model.train(is_train)
    # TODO: complete this function 
    # ===============================================================================
    mean_epoch_loss, mean_iou = 0.0, 0.0
    num_data = len(loader)
    for i, batch in enumerate(loader):
        # origin shape: [4, 3, 240, 320]
        images = batch['input'].to(device)
        masks = batch['target'].to(device)

        # Forward pass
        outputs = model(images)
        # Loss calculation needs rethinking
        loss = criterion(outputs, masks)
        mean_epoch_loss += loss.item()
        
        _, pred = torch.max(outputs, dim=1)
        batch_num = outputs.shape[0]
        class_num = outputs.shape[1]
        temp_iou = 0.0
        for batch_id in range(batch_num):
            temp_iou += np.mean(
                iou(
                    pred[batch_id], 
                    masks[batch_id], 
                    class_num
                    )
            )
        mean_iou += (temp_iou/batch_num)
        

        # Backward and optimze
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 5 == 0:
            print(f'Step[{i+1}/{num_data}, Loss:{loss.item():.4f}]')
    
    mean_epoch_loss /= num_data
    mean_iou /= num_data
    return mean_epoch_loss, mean_iou
    # ===============================================================================

def convert_seg_split_into_color_image(img):
    color_palette = get_tableau_palette()
    colored_mask = np.zeros((*img.shape, 3))

    # print(np.unique(img))

    for i, unique_val in enumerate(np.unique(img)):
        if unique_val == 0:
            obj_color = np.array([0, 0, 0])
        else:
            obj_color = np.array(color_palette[i-1]) * 255
        obj_pixel_indices = (img == unique_val)
        colored_mask[:, :, 0][obj_pixel_indices] = obj_color[0]
        colored_mask[:, :, 1][obj_pixel_indices] = obj_color[1]
        colored_mask[:, :, 2][obj_pixel_indices] = obj_color[2]
    return colored_mask.astype(np.uint8)

def save_learning_curve(train_loss_list, train_miou_list, val_loss_list, val_miou_list):
    """
    In:
        train_loss, train_miou, val_loss, val_miou: list of floats, where the length is how many epochs you trained.
    Out:
        None.
    Purpose:
        Plot and save the learning curve.
    """
    epochs = np.arange(1, len(train_loss_list)+1)
    # plt.figure()
    lr_curve_plot = plt.plot(epochs, train_loss_list, color='navy', label="train_loss")
    plt.plot(epochs, train_miou_list, color='teal', label="train_mIoU")
    plt.plot(epochs, val_loss_list, color='orange', label="val_loss")
    plt.plot(epochs, val_miou_list, color='gold', label="val_mIoU")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.xticks(epochs, epochs)
    plt.yticks(np.arange(10)*0.1, [f"0.{i}" for i in range(10)])
    plt.xlabel('epoch')
    plt.ylabel('mIoU')
    plt.grid(True)
    plt.savefig('learning_curve.png', bbox_inches='tight')

if __name__ == "__main__":
    # ==============Part 4 (a) Training Segmentation model ================
    # Complete all the TODO's in this file
    # - HINT: Most TODO's in this file are exactly the same as homework 2.

    seed(0)
    torch.manual_seed(0)
    # Training setting
    learning_rate = 0.001
    batch_size = 4
    num_epochs = 10
    chkpt_path = 'checkpoint_multi.pth.tar'

    # Check if GPU is being detected
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)


    # TODO: Prepare train and test datasets
    # Load the "dataset" directory using RGBDataset class as a pytorch dataset
    # Split the above dataset into train and test dataset in 9:1 ratio using `torch.utils.data.random_split` method
    # ===============================================================================
    # ===============================================================================
    dataset_dir = "dataset"
    dataset = RGBDataset(dataset_dir)
    train_dataset, test_dataset = random_split(
        dataset=dataset, 
        lengths=[int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)], 
        generator=torch.Generator().manual_seed(42))
    
    # TODO: Prepare train and test Dataloaders. Use appropriate batch size
    # ===============================================================================
    # ===============================================================================
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    # TODO: Prepare model
    # ===============================================================================
    # ===============================================================================
    model = miniUNet(n_channels=3, n_classes=4)
    # TODO: Define criterion, optimizer and learning rate scheduler
    # ===============================================================================
    # ===============================================================================
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # TODO: Train and test the model. 
    # Tips:
    # - Remember to save your model with best mIoU on objects using save_chkpt function
    # - Try to achieve Test mIoU >= 0.9 (Note: the value of 0.9 only makes sense if you have sufficiently large test set)
    # - Visualize the performance of a trained model using save_prediction method. Make sure that the predicted segmentation mask is almost correct.
    # ===============================================================================
    # ===============================================================================
    _train_loss, _train_miou, _test_loss, _test_miou = [], [], [], []
    best_miou = float('-inf')
    for epoch in range(num_epochs):
        print(f'Epoch[{epoch+1}/{num_epochs}]')
        train_loss, train_miou = run(
            model=model, 
            loader=train_loader, 
            criterion=criterion, 
            device=device, 
            is_train=True, 
            optimizer=optimizer)
        test_loss, test_miou = run(
            model=model, 
            loader=test_loader, 
            criterion=criterion, 
            device=device, 
            is_train=False, 
            optimizer=optimizer)
        _train_loss.append(train_loss)
        _train_miou.append(train_miou)
        _test_loss.append(test_loss)
        _test_miou.append(test_miou)
        print('Train loss & mIoU: %0.2f %0.2f' % (train_loss, train_miou))
        print('Validation loss & mIoU: %0.2f %0.2f' % (test_loss, test_miou))
        print('---------------------------------')
        if test_miou > best_miou:
            best_miou = test_miou
            save_chkpt(model, epoch, test_miou, chkpt_path)
    
    # Load the best checkpoint, use save_prediction() on the validation set and test set
    model, epoch, best_miou = load_chkpt(model, chkpt_path, device)
    save_prediction(model, test_loader, dataset_dir+'/pred', device, batch_size)
    save_learning_curve(_train_loss, _train_miou, _test_loss, _test_miou)
        