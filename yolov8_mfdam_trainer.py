import torch
from torch import nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER, DEFAULT_CFG
from mamba_mfdam import MambaMFDAM
import numpy as np

class NeckFeatureExtractor:
    """
    用于从YOLOv8模型中提取neck输出特征的hook管理器
    """
    def __init__(self, model):
        self.model = model
        self.features = []
        self.hooks = []
        # 选择YOLOv8主模型的neck部分
        # 以 backbone 前的 C2f, C2f, SPPF 层为特征输出
        self.neck_layers = self._find_neck_layers()
        self._register_hooks()

    def _find_neck_layers(self):
        # 根据YOLOv8结构，通常4,6,9层传向neck
        layers = []
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'model'):
            for i, m in enumerate(self.model.model.model):
                if i in [4, 6, 9]:  # 取3个尺度输出
                    layers.append(m)
        return layers

    def _hook_fn(self, output):
        # hook回调函数，每次目标层前向传播时自动调用，把输出特征保存到
        self.features.append(output)

    def _register_hooks(self):
        # 在neck_layers每一层上注册forward hook，把hook句柄存入self.hooks
        for layer in self.neck_layers:
            h = layer.register_forward_hook(self._hook_fn)
            self.hooks.append(h)

    def clear(self):
        self.features.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_features(self):
        # 返回neck的三个尺度特征列表
        # 按照 [small, medium, large]
        return [f for f in self.features]

class YOLOv8MFDAMTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        self.domain_weight = overrides.pop('domain_weight', 0.1)
        self.source_data = overrides.pop('source_data', None)
        self.target_data = overrides.pop('target_data', None)
        self.mfdam_module = None
        self.current_epoch = 0
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        neck_channels = self._extract_neck_channels(model)
        if neck_channels:
            self.mfdam_module = MambaMFDAM(
                channels_list=neck_channels,
                num_domains=2,
                alpha_init=0.0
            ).to(self.device)
            LOGGER.info(f"初始化Mamba-MFDAM，通道数: {neck_channels}")
        else:
            LOGGER.warning("未能提取neck通道数，MFDAM模块未初始化")
        return model

    def get_dataloader(self, dataset_path, mode, batch_size):
        # 复用YOLOv8原始dataloader逻辑
        from ultralytics.models.yolo.detect.train import DetectionTrainer as BaseTrainer
        base_trainer = BaseTrainer(self.args)
        return base_trainer.get_dataloader(dataset_path, mode, batch_size)

    def extract_neck_features(self, imgs):
        """利用hook机制提取neck特征"""
        self.neck_extractor.clear()
        _ = self.model(imgs)  # 前向传播，hook自动保存特征
        feats = self.neck_extractor.get_features()
        # 确保返回3尺度特征
        if len(feats) == 3:
            return feats
        else:
            # 回退为虚拟特征
            batch_size = imgs.size(0)
            LOGGER.warning("Neck特征数缺失，使用虚拟neck特征！")
            return [
                torch.randn(batch_size, 256, 20, 20).to(self.device),
                torch.randn(batch_size, 512, 10, 10).to(self.device),
                torch.randn(batch_size, 1024, 5, 5).to(self.device)
            ]

    def train(self):
        source_loader = self.get_dataloader(self.source_data, 'train', self.args.batch)
        target_loader = self.get_dataloader(self.target_data, 'train', self.args.batch)
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        num_batches = min(len(source_loader), len(target_loader))

        # 初始化neck特征提取器
        self.neck_extractor = NeckFeatureExtractor(self.model)

        for epoch in range(self.epochs):
            self.model.train()
            for batch_idx in range(num_batches):
                # 源域batch
                try:
                    source_batch = next(source_iter)
                except StopIteration:
                    source_iter = iter(source_loader)
                    source_batch = next(source_iter)
                source_imgs = source_batch['img'].to(self.device)
                source_domain_labels = torch.zeros(source_imgs.size(0), dtype=torch.long, device=self.device)  # 源域=0

                # 1. 检测损失
                preds = self.model(source_imgs)
                yolo_loss = super().loss(source_batch, preds)
                # 2. neck特征提取
                neck_features = self.extract_neck_features(source_imgs)
                # 3. 域判别损失
                _, domain_loss = self.mfdam_module(neck_features, source_domain_labels)
                total_loss = yolo_loss + self.domain_weight * domain_loss if domain_loss is not None else yolo_loss
                total_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # 目标域batch
                try:
                    target_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    target_batch = next(target_iter)
                target_imgs = target_batch['img'].to(self.device)
                target_domain_labels = torch.ones(target_imgs.size(0), dtype=torch.long, device=self.device)  # 目标域=1

                # 只做域判别损失，不做检测损失
                neck_features = self.extract_neck_features(target_imgs)
                _, domain_loss = self.mfdam_module(neck_features, target_domain_labels)
                if domain_loss is not None:
                    domain_loss = self.domain_weight * domain_loss
                    domain_loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            self.current_epoch += 1
            LOGGER.info(f"完成第{epoch+1}轮训练")

        # 训练结束后移除hook
        self.neck_extractor.remove()

    def _extract_neck_channels(self, model):
        neck_channels = []
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'model'):
                for i, module in enumerate(model.model.model):
                    if hasattr(module, 'cv2') and hasattr(module.cv2, 'conv'):
                        if i >= 12:
                            neck_channels.append(module.cv2.conv.out_channels)
                    elif hasattr(module, 'c2') and hasattr(module.c2, 'conv'):
                        if i >= 12:
                            neck_channels.append(module.c2.conv.out_channels)
            if not neck_channels:
                neck_channels = [256, 512, 1024]
        except Exception as e:
            LOGGER.warning(f"提取neck通道时出错: {e}，使用默认值")
            neck_channels = [256, 512, 1024]
        return neck_channels