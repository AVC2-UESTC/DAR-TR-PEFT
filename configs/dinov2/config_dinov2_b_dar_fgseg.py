
_base_ = [
    '../_base_/datasets/FGSeg_cfg.py',
    '../_base_/schedules/temp_scheduler.py', 
    # '../_base_/models/CaMP_fgseg.py'  
]


runtime = dict(
    logger_name = 'default',
    
)


Dataset_cfg = dict(
    dataset_cfg_args=dict(
        # data_root=['./datasets/CAMO', './datasets/COD10KCAM'], 
        # data_root = './datasets/DUTS',
        data_root='./datasets/DUTS', 
        # data_root='./datasets/CHAMELEON', 
        # data_root='./datasets/COD10KCAM', 
        # data_root='./datasets/KvasirSEG',
        # data_root='./datasets/ISIC2017',
        # seg_map_suffix = '.jpg', 
        # img_suffix = '.png',
        truncate_ratio=None,
        transform_cfg_args=dict(
            
            train_pipeline=[
                dict(
                    transform_type='ToTensor', 
                ), 
                dict(
                    transform_type='RandomResize',
                    scale=(518, 518),
                    ratio_range=(1, 1.333),
                    keep_ratio=False,
                    resize_mask=True
                ), 
                dict(
                    transform_type='RandomCrop',
                    size=518
                ), 
                dict(
                    transform_type='RandomHorizontalFlip',
                    prob=0.5
                ), 
                dict(
                    transform_type='Normalize',
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            ], 
            validate_pipeline=[
                dict(
                    transform_type='ToTensor',
                ), 
                dict(
                    transform_type='Resize',
                    scale=(682, 518),
                    keep_ratio=True,
                    resize_mask=True
                ), 
                dict(
                    transform_type='CenterCrop',
                    size=518
                ), 
                dict(
                    transform_type='Normalize',
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            ]
            # validate_pipeline=[
            #     dict(
            #         transform_type='ToTensor',
            #     ), 
            #     dict(
            #         transform_type='Resize',
            #         scale=(518, 518),
            #         keep_ratio=False,
            #         resize_mask=True
            #     ), 
            #     dict(
            #         transform_type='Normalize',
            #         mean=(0.485, 0.456, 0.406),
            #         std=(0.229, 0.224, 0.225)
            #     )
            # ]
            
            
        )
    )
)

warmup = 10
epochs = 30
Scheduler_cfg = dict(
    scheduler_cfg_args=dict(
        device='cuda',
        seed=3407,
        
        num_workers=0,
        
        optimizer_name='AdamW',
        optimizer_args=dict(
                    lr=0.00015,
                    weight_decay=1e-4,
                ),
        
        lr_scheduler_name= 'CosineAnnealingLR_Warmup',
        lr_scheduler_args= dict(
                    epochs=epochs,     
                    warmup_epochs=warmup,
                    warmup_factor=1e-3, 
                    end_factor=1e-6
                ),
        
        batch_size=10,
        epochs=epochs,
        eval_interval=epochs + 1, # no validation during training
        amp = False,
   
        metrics_cfg=[
            dict(metric_type='MAE', 
                 resize_logits=True, 
                 name='MAE'),
            
            # dict(metric_type='Fmeasure', 
            #      resize_logits=True, 
            #      name='adp_F', 
            #      metric_args=dict(
            #         beta_sq=0.3, 
            #         mode='adaptive')),
            
            dict(metric_type='WeightedFmeasure', 
                 resize_logits=True, 
                 name='Weighted_F', 
                 metric_args=dict(
                     beta_sq=0.3)),
            
            dict(metric_type='Smeasure', 
                 resize_logits=True, 
                 name='S_measure', 
                 metric_args=dict(
                     alpha=0.5)),
            
            dict(metric_type='Emeasure', 
                 resize_logits=True, 
                 name='E_measure', 
                 metric_args=dict(
                     mode='mean')), 
            
            # dict(metric_type='Fmeasure', 
            #      resize_logits=True, 
            #      name='max_F', 
            #      metric_args=dict(
            #         beta_sq=0.3, 
            #         mode='max')),
            
            # dict(metric_type='Emeasure', 
            #      resize_logits=True, 
            #      name='max_Emeasure', 
            #      metric_args=dict(
            #          mode='max')),
        ]
    )
)


Model_cfg = dict(
    model_cfg_name="DinoVisionTransformer_DAR_Seg_Config",
    model_cfg_args=dict(
        
        pretrained_weights = './model_weights/dinov2_vitb14_pretrain_new.pth',
        finetune_weights = './model_weights/dinov2_b_dar_duts.pth',
    

        tuning_mode = 'PEFT',
        
        backbone_cfg=dict(
            img_size=518, # 518
            patch_size=14,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            num_register_tokens=0, # reg tokens does not have pos_embed
            interpolate_antialias=True,
            interpolate_offset=0.0,
            block_chunks=0,
            init_values=1e-5,
            ft_cfg=dict(
                bottleneck=[72, 48],
                adapter_scalar=2.0,
                learnable_scalar=True,
                act_cfg=dict(
                    act_type="ReLU",
                    layer_args=dict(
                        inplace=True
                    )
                ),
                pt_act_cfg=dict(act_type='GELU'),
                
                adapter_layernorm_option=None,
                dropout_layer=dict(
                    drop_type="Dropout",
                    drop_prob=0.0,
                    inplace=True
                )
            ),
                
        ),
        decode_head_cfg = dict(
            in_channels=[768,],#
            channels=256,
            num_classes=2,
            out_channels=1,
                
            norm_cfg=dict(
                norm_type='BatchNorm2d',
                requires_grad=True,
                layer_args=dict(
                    eps=1e-5, 
                    momentum=0.1, 
                    affine=True,
                    track_running_stats=True
                )
            ),
                
            in_index=[0,],  
            align_corners=False,
            
            resize_ratio=4
        ),
        
        threshold=None,
        loss_decode=[
            dict(
                loss_type="BCEWithLogitsLoss",
                reduction="mean",
                loss_weight=1.0,
                loss_name="mask_loss_bce"
            ),
            
            dict(
                loss_type="DiceLoss",
                reduction="mean",
                loss_weight=0.5,
                loss_name="mask_loss_dice"
            ), 
            
            dict(
                loss_type='AdaLoss', 
                reduction="mean",
                loss_weight=2.0,
                loss_name='tokenreg_loss_adaloss',
                loss_args=dict(
                    token_target_ratio=0.5
                )
            ),
            
            
            
            # dict(
            #     loss_type="UALLoss",
            #     reduction="mean",
            #     loss_weight=1.0,
            #     loss_name='maskreg_loss_ual'
            # ),
            
            # dict(
            #     loss_type='CosSimilarityLoss',
            #     reduction="mean",
            #     loss_weight=0.5,
            #     loss_name="alg_loss_cos"
            # )
        ]
    )
)
