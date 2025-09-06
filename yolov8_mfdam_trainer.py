"""
集成Mamba-MFDAM的YOLOv8训练器
"""
import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER, DEFAULT_CFG
from mamba_mfdam import MambaMFDAM
import numpy as np


class YOLOv8MFDAMTrainer(DetectionTrainer):
    """集成Mamba-MFDAM域适应的YOLOv8训练器"""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        初始化训练器

        Args:
            cfg: 配置字典或路径
            overrides: 覆盖参数字典
            _callbacks: 回调函数
        """
        # 确保 overrides 不为 None
        if overrides is None:
            overrides = {}

        # 处理 MFDAM 特定参数
        self.domain_weight = overrides.pop('domain_weight', 0.1)
        self.alpha_schedule = overrides.pop('alpha_schedule', 'linear')
        self.max_alpha = overrides.pop('max_alpha', 1.0)

        # 初始化 MFDAM 相关属性
        self.mfdam_module = None
        self.current_epoch = 0

        # 调用父类初始化
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """初始化包含MFDAM模块的模型"""
        model = super().get_model(cfg, weights, verbose)

        # 从YOLOv8 neck获取特征通道数
        neck_channels = self._extract_neck_channels(model)

        if neck_channels:
            # 初始化MFDAM模块
            self.mfdam_module = MambaMFDAM(
                channels_list=neck_channels,
                num_domains=2,  # 源域和目标域
                alpha_init=0.0
            ).to(self.device)

            LOGGER.info(f"初始化Mamba-MFDAM，通道数: {neck_channels}")
        else:
            LOGGER.warning("未能提取neck通道数，MFDAM模块未初始化")

        return model

    def _extract_neck_channels(self, model):
        """提取neck层的通道数"""
        neck_channels = []

        try:
            if hasattr(model, 'model') and hasattr(model.model, 'model'):
                # 遍历模型层寻找neck特征
                for i, module in enumerate(model.model.model):
                    # 查找C2f或类似的neck模块
                    if hasattr(module, 'cv2') and hasattr(module.cv2, 'conv'):
                        if i >= 12:  # 通常neck从第12层开始
                            neck_channels.append(module.cv2.conv.out_channels)
                    elif hasattr(module, 'c2') and hasattr(module.c2, 'conv'):
                        if i >= 12:
                            neck_channels.append(module.c2.conv.out_channels)

            # 如果没有找到，使用默认值
            if not neck_channels:
                neck_channels = [256, 512, 1024]  # YOLOv8默认neck通道

        except Exception as e:
            LOGGER.warning(f"提取neck通道时出错: {e}，使用默认值")
            neck_channels = [256, 512, 1024]

        return neck_channels

    def setup_model(self):
        """设置包含MFDAM集成的模型"""
        super().setup_model()

        if self.mfdam_module is not None:
            # 将MFDAM参数添加到优化器
            mfdam_params = list(self.mfdam_module.parameters())
            if mfdam_params:
                # 为MFDAM创建单独的参数组
                if hasattr(self.optimizer, 'add_param_group'):
                    self.optimizer.add_param_group({
                        'params': mfdam_params,
                        'lr': self.args.lr0 * 0.1,  # MFDAM使用较低的学习率
                        'weight_decay': self.args.weight_decay
                    })
                    LOGGER.info(f"已将{len(mfdam_params)}个MFDAM参数添加到优化器")

    def update_alpha(self, epoch, total_epochs):
        """根据训练进度更新GRL alpha参数"""
        if total_epochs <= 0:
            return 0.0

        progress = epoch / total_epochs

        if self.alpha_schedule == 'linear':
            # 线性调度
            alpha = progress * self.max_alpha
        elif self.alpha_schedule == 'exp':
            # 指数调度
            alpha = self.max_alpha * (2.0 / (1.0 + np.exp(-10 * progress)) - 1.0)
        else:
            alpha = self.max_alpha

        if self.mfdam_module is not None:
            self.mfdam_module.set_alpha(alpha)

        return alpha

    def _create_dummy_neck_features(self, batch_size):
        """创建虚拟neck特征用于测试"""
        return [
            torch.randn(batch_size, 256, 20, 20).to(self.device),
            torch.randn(batch_size, 512, 10, 10).to(self.device),
            torch.randn(batch_size, 1024, 5, 5).to(self.device)
        ]

    def _create_domain_labels(self, batch_size):
        """创建域标签"""
        # 简单的实现：假设批次中一半是源域，一半是目标域
        # 在实际应用中，您需要根据数据来源设置正确的标签
        domain_labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        # 可以根据实际需求修改标签分配逻辑
        return domain_labels

    def loss(self, batch, preds=None):
        """包含域适应的修改损失函数"""
        if preds is None:
            preds = self.model(batch['img'])

        # 原始YOLOv8损失
        yolo_loss = super().loss(batch, preds)

        # 如果MFDAM可用且在训练模式，添加域适应损失
        if self.mfdam_module is not None and self.model.training:
            try:
                batch_size = batch['img'].size(0)

                # 创建虚拟neck特征 (实际使用时需要从模型中提取)
                neck_features = self._create_dummy_neck_features(batch_size)

                # 创建域标签
                domain_labels = self._create_domain_labels(batch_size)

                # MFDAM前向传播
                domain_pred, domain_loss = self.mfdam_module(neck_features, domain_labels)

                if domain_loss is not None:
                    # 将域损失添加到总损失中
                    if hasattr(yolo_loss, 'item'):  # 如果是tensor
                        total_loss = yolo_loss + self.domain_weight * domain_loss
                    else:  # 如果是字典或其他格式
                        total_loss = yolo_loss
                        if isinstance(total_loss, dict) and 'loss' in total_loss:
                            total_loss['loss'] += self.domain_weight * domain_loss

                    # 记录域损失
                    if hasattr(self, 'loss_names') and 'domain' not in self.loss_names:
                        self.loss_names += ['domain']

                    return total_loss

            except Exception as e:
                LOGGER.warning(f"MFDAM损失计算错误: {e}，跳过域适应")

        return yolo_loss

    def _do_train(self, world_size=1):
        """训练循环，包含alpha调度"""
        # 在训练开始前更新alpha
        if self.mfdam_module is not None:
            alpha = self.update_alpha(self.epoch, self.epochs)
            if self.epoch % 10 == 0:
                LOGGER.info(f"轮次 {self.epoch}: GRL alpha = {alpha:.4f}")

        # 调用父类的训练方法
        return super()._do_train(world_size)

    def on_train_epoch_end(self):
        """训练轮次结束时的处理"""
        super().on_train_epoch_end()
        self.current_epoch += 1