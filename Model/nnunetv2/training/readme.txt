nnUNetTrainer1：常规训练模型
nnUNetTrainer2：协同优化
    nnUNetTrainer0：双分支训练
    nnUNetTrainer1：#双分支，但是共享上采样
    nnUNetTrainer2：#上采样不共享+优化
    nnUNetTrainer3：#上采样最后一个不共享（已经修改为nnUNetTrainer0+第一级编码器不共享）
    nnUNetTrainerDNet：D-Net与MedNeXt胡乱结合，效果没有获得提升，这是中间尝试
    nnUNetTrainerSVSUP：3D
    nnUNetTrainerDBUNet：3D
    nnUNetTrainerI2UNet：2D
    nnUNetTrainerMTFormer：2D
    nnUNetTrainerMTLSegFormer：2D
    nnUNetTrainerMulT：2D
    nnUNetTrainerDBSNet0 #medxnet双分支！！！！消融1-基线
    nnUNetTrainerDBSNet4 #简单上采样。相对于nnUNetTrainerDBSNet0有效。消融2-说明简化上采样性能不会有太多损失
    nnUNetTrainerDBSNet3：##简单上采样+注意力 CBAM，相对于4有提升。消融4-信息交互有必要，但是简单相加并不是特别有效
    nnUNetTrainerDBSNet44：#简单上采样+通道拼接融合。可作为消融3，证明信息交互有效
    nnUNetTrainerDBSNet6 #简单上采样+EfficientCrossAttention交叉注意力，有一定性能提升
    nnUNetTrainerDBSNet66 #0.6+0.4；nnUNetTrainerDBSNet6f
    nnUNetTrainerDBSNet666 #0.7+0.3，finetune
    nnUNetTrainerDBSNet6666 #0.4+0.6，finetune
    nnUNetTrainerDBSNet66666 #0.3+0.7，finetune
