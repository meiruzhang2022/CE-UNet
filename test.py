import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
devicess = [0]
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import csv
import argparse
import numpy as np
from PIL import Image
from thop import profile, clever_format
import importlib
import torch
from torch.utils.data import DataLoader
import torchio
from torchio.transforms import ZNormalization
from tqdm import tqdm
import pandas as pd
from hparam import hparams as hp
from utils.metric import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from data_function import MedData_val
from loss_function import Binary_Loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

source_test_dir = hp.source_test_dir
label_test_dir = hp.label_test_dir

source_val_dir = hp.source_val_dir
label_val_dir = hp.label_val_dir

output_dir_test = hp.output_dir_test

def parse_training_args(parser):
    """
    Parse commandline arguments.
    """

    parser.add_argument('-o', '--output_dir', type=str, default=hp.output_dir, required=False,help='Directory to save checkpoints')
    parser.add_argument('-out', '--output_dir_test', type=str, default=hp.output_dir_test, required=False,help='Directory to save predictions')
    parser.add_argument('--latest-checkpoint-file', type=str, default=hp.latest_checkpoint_file,help='Store the latest checkpoint in each epoch')
    parser.add_argument('--best_dice_model_file', type=str, default=hp.best_dice_model_file,help='Store the best_dice_model checkpoint in each epoch')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, default=hp.total_epochs, help='Number of total epochs to run')
    training.add_argument('--epochs-per-checkpoint', type=int, default=hp.epochs_per_checkpoint,help='Number of epochs per checkpoint')
    training.add_argument('--batch', type=int, default=hp.batch_size, help='batch-size')
    training.add_argument('--best_dice', type=int, default=hp.best_dice, help='best-dice')
    parser.add_argument('-k',"--ckpt",type=str,default=hp.ckpt,help="path to the checkpoints to resume training")
    parser.add_argument("--init-lr", type=float, default=hp.init_lr, help="learning rate")
    # TODO
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
    training.add_argument('--amp-run', action='store_true', help='Enable AMP')
    training.add_argument('--cudnn-enabled', default=True, help='Enable cudnn')
    training.add_argument('--cudnn-benchmark', default=True, help='Run cudnn benchmark')
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true',help='disable uniform initialization of batchnorm layer weight')
    
    return parser
def validate(model):
    parser = argparse.ArgumentParser(description='PyTorch Image Segmentation Validation')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    # 加载模型权重
    ckpt_path = os.path.join(args.output_dir, args.best_dice_model_file)
    print("加载模型权重:", ckpt_path)
    model = torch.nn.DataParallel(model, device_ids=devicess).cuda()
    ckpt = torch.load(ckpt_path, map_location='cuda')
    model.load_state_dict(ckpt["model"])
    model.eval()

    criterion = Binary_Loss().cuda()
    val_dataset = MedData_val(source_val_dir, label_val_dir)
    val_loader = DataLoader(val_dataset.queue_dataset, batch_size=8, num_workers=4,shuffle=False, pin_memory=True, drop_last=False)

    print("开始验证...")
    val_loss = val_accuracy = val_dice = val_jaccard = val_precision = val_recall = val_specificity = 0
    num_images = 0

    per_image_metrics = []
    save_dir = os.path.join(args.output_dir_test, "predict_masks")
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(val_loader):
            x = batch['source']['data'].float().cuda()
            y = batch['label']['data'].float().cuda()

            if hp.mode == '2d':
                x = x.squeeze(4)
                y = y.squeeze(4)
                y[y != 0] = 1

            outputs = model(x)
            loss = criterion(outputs, y)

            logits = torch.sigmoid(outputs)
            labels = (logits > 0.5).float()

            B = x.size(0)
            for b in range(B):
                pred_mask_np = labels[b, 0].cpu().numpy() * 255

                # 自动获取原始名字（需要在 MedData_val 的 Subject 中加 name）
                if 'name' in batch:
                    name = batch['name'][b] + ".png"
                else:
                    name = f"img_{num_images:04d}.png"

                pred_img = Image.fromarray(pred_mask_np.astype('uint8'))
                pred_img.save(os.path.join(save_dir, name))

                acc, dsc, jac, rec, prec, spe = metric(y[b].cpu(), labels[b].cpu())
                per_image_metrics.append({
                    "image": name,
                    "accuracy": acc,
                    "dice": dsc,
                    "jaccard": jac,
                    "recall": rec,
                    "precision": prec,
                    "specificity": spe,
                })

                val_accuracy += acc
                val_dice += dsc
                val_jaccard += jac
                val_precision += prec
                val_recall += rec
                val_specificity += spe
                val_loss += loss.item() / B
                num_images += 1

    # 计算平均
    val_loss /= num_images
    val_accuracy /= num_images
    val_dice /= num_images
    val_jaccard /= num_images
    val_precision /= num_images
    val_recall /= num_images
    val_specificity /= num_images

    # 保存指标
    csv_path = os.path.join(args.output_dir_test, "val_metrics.csv")
    pd.DataFrame(per_image_metrics).to_csv(csv_path, index=False)
    print("✅ 每张图像指标保存至:", csv_path)
    print(f"✅ 平均验证结果 - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Dice: {val_dice:.4f}, "
          f"Jaccard: {val_jaccard:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, Specificity: {val_specificity:.4f}")
def set_seed(seed):
    """
    设置随机数种子，包括 CPU 和 GPU。
    """
    torch.manual_seed(seed)  # 设置 PyTorch 的随机数种子
    torch.cuda.manual_seed(seed)  # 设置 GPU 的随机数种子
    torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的随机数种子
    torch.backends.cudnn.deterministic = True  # 确保结果的可重复性
    torch.backends.cudnn.benchmark = False  # 关闭 cuDNN 的优化，确保结果一致
    
def load_model(model_name):
    """根据模型名称动态加载对应的 U-Net 模型"""
    try:
        # 动态导入模块
        module = importlib.import_module(f"models.two_d.{model_name}")
        model = module.Unet(in_channels=1, classes=1)
        return model
    except ImportError:
        raise NotImplementedError(f"模型 {model_name} 未实现或导入失败")

if __name__ == '__main__':
    set_seed(42)
    #total
    #model_names=['MRFusion','MRFusion3','MRFusion3_avg','Unet','UNETR','AE_Unet','segnet','UNETplusplus','deeplab','swin_unet_modify','TransUNet']
    #1
    model_names=['MRFusion','MRFusion3_avg','Unet','UNETR','AE_Unet','segnet','UNETplusplus','miniseg','MRFusion3',]
    model_names=['MRFusion3']
    for model_name in model_names:
        print(model_name)
        model=load_model(model_name)
        #continue
        hp.output_dir='logs1'
        hp.output_dir = os.path.join(hp.output_dir, model_name)
        print(hp.output_dir)
        hp.output_dir_test="results1t"
        hp.output_dir_test= os.path.join(hp.output_dir_test, model_name)
        hp.ckpt = os.path.join(hp.output_dir, hp.latest_checkpoint_file)
        hp.train_or_test='test'
        if hp.train_or_test == 'test':
            validate(model)  # 替换掉 test(model)
