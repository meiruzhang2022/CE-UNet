class hparams:

    train_or_test = 'train'
    #train_or_test = 'test'
    output_dir = 'logs3'
    train_aug = True
    val_aug=False
    latest_checkpoint_file = 'checkpoint_latest.pt'
    best_dice_model_file='best_dice_model.pt'
    total_epochs = 50
    epochs_per_checkpoint = 1
    batch_size = 8
    #ckpt = 'log_thyroid/Unet_SKD/'
    ckpt=None
    init_lr = 0.0001
    scheduer_step_size = 20
    scheduer_gamma = 0.8
    debug = False
    mode = '2d' # '2d or '3d'
    in_class = 1
    out_class = 1
    best_dice=0.0
    
    crop_or_pad_size =256,256,1 # if 2D: 256,256,1
    patch_size = 256,256,1 # if 2D: 128,128,1 

    # for test
    patch_overlap = 4,4,0 # if 2D: 4,4,0

    fold_arch = '*.nii'

    save_arch = '.png'

    source_train_dir = './thyroid-1/Train/images'
    label_train_dir =  './thyroid-1/Train/masks'
    source_val_dir = './thyroid-1/Test/images'
    label_val_dir = './thyroid-1/Test/masks'
    source_test_dir = './thyroid-1/Test/images'
    label_test_dir = './thyroid-1/Test/masks'
    
    # source_train_dir = './thyroid-5/Test/images'
    # label_train_dir =  './thyroid-5/Test/masks'
    # source_val_dir = './thyroid-5/Train/images'
    # label_val_dir = './thyroid-5/Train/masks'
    # source_test_dir = './thyroid-5/Train/images'
    # label_test_dir = './thyroid-5/Train/masks'
    # output_dir_test = 'results/thyroid5'
    output_dir_test = 'results/thyroid-1'