import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
devicess = [0]
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import numpy as np
from PIL import Image
from thop import profile, clever_format
import importlib
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchio
from torchio.transforms import ZNormalization
from tqdm import tqdm
import pandas as pd
from hparam import hparams as hp
from utils.metric import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from loss_function import DiceLoss,Binary_Loss,FocalLoss,BCEDiceLoss
from data_function import MedData_train, MedData_test,MedData_val

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

source_train_dir = hp.source_train_dir
label_train_dir =  hp.label_train_dir

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
    
def train(model):
    parser = argparse.ArgumentParser(description='PyTorch Image Segmentation Training')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    
    os.makedirs(args.output_dir, exist_ok=True)

    model = torch.nn.DataParallel(model, device_ids=devicess)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)    
    scheduler = StepLR(optimizer, step_size=hp.scheduer_step_size, gamma=hp.scheduer_gamma)

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        print(os.path.join(args.output_dir, args.latest_checkpoint_file))
        ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file),
                          map_location=lambda storage, loc: storage)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        scheduler.load_state_dict(ckpt["scheduler"])
        elapsed_epochs = ckpt["epoch"]
    else:
        elapsed_epochs = 0
    best_dice=args.best_dice    
    model.cuda()    
    criterion =  Binary_Loss().cuda()
    writer = SummaryWriter(args.output_dir)
    
    train_dataset = MedData_train(source_train_dir, label_train_dir)
    train_loader = DataLoader(train_dataset.queue_dataset,batch_size=args.batch,shuffle=True,
            num_workers=4,pin_memory=True,prefetch_factor=8,drop_last=True)
    model.train()
    epochs = args.epochs - elapsed_epochs
    iteration = elapsed_epochs * len(train_loader)

    for epoch in range(1, epochs + 1):
        print("epoch:" + str(epoch))
        epoch += elapsed_epochs
        num_iters = 0
        for i, batch in enumerate(train_loader):
            if hp.debug:
                if i >= 1:
                    break

            optimizer.zero_grad()

            if (hp.in_class == 1) and (hp.out_class == 1):
                x = batch['source']['data']
                y = batch['label']['data']

                x = x.type(torch.FloatTensor).cuda()
                y = y.type(torch.FloatTensor).cuda()

            if hp.mode == '2d':
                x = x.squeeze(4)
                y = y.squeeze(4)
                y[y != 0] = 1

            outputs = model(x)
            logits = torch.sigmoid(outputs)
            labels = logits.clone()
            labels[labels > 0.5] = 1
            labels[labels <= 0.5] = 0            
            loss = criterion(outputs, y)            
            num_iters += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            iteration += 1
        scheduler.step()        
        torch.save({"model": model.state_dict(),"optim": optimizer.state_dict(),"scheduler": scheduler.state_dict(),"epoch": epoch},
            os.path.join(args.output_dir, args.latest_checkpoint_file))

        if epoch % args.epochs_per_checkpoint == 0:
            model.eval()
            val_dataset = MedData_val(source_val_dir, label_val_dir)
            val_loader = DataLoader(val_dataset.queue_dataset,batch_size=1,num_workers=0,shuffle=False,pin_memory=True,drop_last=False)
            num_val_iters,val_loss, val_accuracy,val_dice,val_jaccard, val_precision, val_recall, val_specificity=0,0,0,0,0,0,0,0
            for i, batch in enumerate(val_loader):
                if hp.debug:
                    if i >= 1:
                        break
                x = batch['source']['data'].type(torch.FloatTensor).cuda()
                y = batch['label']['data'].type(torch.FloatTensor).cuda()

                x = x.squeeze(4)
                y = y.squeeze(4)
                y[y != 0] = 1

                with torch.no_grad():
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    logits = torch.sigmoid(outputs)
                    labels = logits.clone()
                    labels[labels > 0.5] = 1
                    labels[labels <= 0.5] = 0
                val_loss += loss.item()
                accurary,dice,jaccard,precision,recall,specificity = metric(y.cpu(), labels.cpu())
                val_accuracy += accurary
                val_dice += dice
                val_jaccard += jaccard
                val_precision += precision
                val_recall += recall
                val_specificity += specificity                
                num_val_iters += 1
            val_loss /= num_val_iters
            val_accuracy/= num_val_iters
            val_dice /= num_val_iters
            val_jaccard /= num_val_iters
            val_precision /= num_val_iters
            val_recall /= num_val_iters
            val_specificity /= num_val_iters
            
            writer.add_scalar('Validation/val_Accuracy', val_accuracy, epoch)
            writer.add_scalar('Validation/val_Dice', val_dice, epoch)
            writer.add_scalar('Validation/val_Jaccard', val_jaccard, epoch)
            writer.add_scalar('Validation/val_Loss', val_loss, epoch)
            writer.add_scalar('Validation/val_Precision', val_precision, epoch)
            writer.add_scalar('Validation/val_Recall', val_recall, epoch)
            writer.add_scalar('Validation/val_Specificity',  val_specificity, epoch)
            print("Validation Loss:", val_loss,"Validation Dice:", val_dice)
            # if val_dice > best_dice:
            #     print(f"Dice improved from {best_dice:.4f} to {val_dice:.4f}. Saving best model...")
            #     best_dice = val_dice
            #     torch.save(
            #         {
            #             "model": model.state_dict(),
            #             "optim": optimizer.state_dict(),
            #             "epoch": epoch,
            #         },
            #         os.path.join(args.output_dir, f"best_dice_model{epoch}.pt"),
            #     )
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(args.output_dir, f"best_dice_model{epoch}.pt"),
                )

    writer.close()



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
    model_names=['MRFusion','MRFusion3','MRFusion3_avg']
    
    #2
    #model_names=['MRFusion','MRFusion3']
    #3
    #model_names=['segnet','UNETplusplus','MRFusion3_std','MRFusion','MRFusion3','deeplab','swin_unet_modify','TransUNet']
    #4
    #model_names=['MRFusion','MRFusion3','MRFusion3_avg','Unet','UNETR','AE_Unet','UNETplusplus','swin_unet_modify','TransUNet']
    #5
    #model_names=['MRFusion','MRFusion3','swin_unet_modify','TransUNet']
    for model_name in model_names:
        print(model_name)
        model=load_model(model_name)
        #continue
        hp.output_dir='logs1'
        hp.output_dir = os.path.join(hp.output_dir, model_name)
        hp.output_dir_test= os.path.join(hp.output_dir_test, model_name)
        train(model)

