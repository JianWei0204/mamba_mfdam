import torch
import yaml
import os
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER, DEFAULT_CFG
from torch.utils.data import DataLoader
from mamba_mfdam import MambaMFDAM
from neck_feature_extractor import NeckFeatureExtractor


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

    def get_backbone_params(self):
        backbone_layers = [self.model.model[i] for i in range(10)]  # 前10层为backbone
        backbone_params = []
        for layer in backbone_layers:
            backbone_params += list(layer.parameters())
        return backbone_params

    def build_optimizer_domain(self, lr=0.001, weight_decay=0.0005, momentum=0.9):
        backbone_params = self.get_backbone_params()
        mfdam_params = list(self.mfdam_module.parameters())
        params = backbone_params + mfdam_params
        optimizer_domain = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        return optimizer_domain

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        # 构建域适应优化器
        self.optimizer_domain = self.build_optimizer_domain(
            lr=self.args.lr0, weight_decay=self.args.weight_decay, momentum=self.args.momentum
        )

    def _do_train(self, world_size=1):
        """融合YOLOv8标准训练+自定义域判别训练"""
        try:
            self._setup_train(world_size)
            self.neck_extractor = NeckFeatureExtractor(self.model)
            self.validator = self.get_validator()

            source_dataset = self.build_dataset(self.source_data["train"], mode='train', batch=self.args.batch)
            target_dataset = self.build_dataset(self.target_data["train"], mode='train', batch=self.args.batch)
            source_loader = DataLoader(source_dataset, batch_size=self.args.batch, shuffle=True,
                                       num_workers=self.args.workers,
                                       collate_fn=source_dataset.collate_fn if hasattr(source_dataset,
                                                                                       'collate_fn') else None)
            target_loader = DataLoader(target_dataset, batch_size=self.args.batch, shuffle=True,
                                       num_workers=self.args.workers,
                                       collate_fn=target_dataset.collate_fn if hasattr(target_dataset,
                                                                                       'collate_fn') else None)

            source_iter = iter(source_loader)
            target_iter = iter(target_loader)
            num_batches = min(len(source_loader), len(target_loader))

            LOGGER.info(f"开始交替训练: 源域{len(source_loader)}批次, 目标域{len(target_loader)}批次")

            # ==== 标准YOLOv8训练日志、权重与验证相关初始化 ====
            best_fitness = -1
            weights_dir = os.path.join(self.save_dir, "weights")
            os.makedirs(weights_dir, exist_ok=True)
            results_csv = os.path.join(self.save_dir, "results.csv")
            with open(results_csv, "w") as f:
                f.write("epoch,train_loss,val_map50\n")

            # ==== 训练循环 ====
            for epoch in range(self.epochs):
                print(f"Epoch {epoch + 1} start, conv0 mean: {self.model.model[0].conv.weight.mean().item():.6f}")
                self.model.train()
                epoch_loss = 0.0

                for batch_idx in range(num_batches):
                    # 源域batch训练
                    try:
                        source_batch = next(source_iter)
                    except StopIteration:
                        source_iter = iter(source_loader)
                        source_batch = next(source_iter)
                    source_imgs = source_batch['img']
                    if source_imgs.dtype == torch.uint8:
                        source_imgs = source_imgs.float() / 255.0
                    source_imgs = source_imgs.to(self.device)
                    source_batch['img'] = source_imgs
                    source_domain_labels = torch.zeros(source_imgs.size(0), dtype=torch.long, device=self.device)
                    # 检测损失
                    loss, loss_items = self.model(source_batch)
                    self.scaler.scale(loss.sum()).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    epoch_loss += loss.sum().item()

                    # 目标域batch训练
                    try:
                        target_batch = next(target_iter)
                    except StopIteration:
                        target_iter = iter(target_loader)
                        target_batch = next(target_iter)
                    target_imgs = target_batch['img']
                    if target_imgs.dtype == torch.uint8:
                        target_imgs = target_imgs.float() / 255.0
                    target_imgs = target_imgs.to(self.device)
                    target_domain_labels = torch.ones(target_imgs.size(0), dtype=torch.long, device=self.device)
                    # 域判别损失
                    neck_features_src = self.extract_neck_features(source_imgs)
                    _, domain_loss_src = self.mfdam_module(neck_features_src, source_domain_labels)
                    neck_features_tgt = self.extract_neck_features(target_imgs)
                    _, domain_loss_tgt = self.mfdam_module(neck_features_tgt, target_domain_labels)
                    domain_loss = self.domain_weight * (domain_loss_src + domain_loss_tgt)
                    self.scaler.scale(domain_loss.sum()).backward()
                    self.scaler.step(self.optimizer_domain)
                    self.scaler.update()
                    self.optimizer_domain.zero_grad()

                self.current_epoch += 1
                avg_loss = epoch_loss / num_batches

                # ==== 标准YOLOv8验证与权重保存 ====
                print(f"Epoch {epoch + 1} end, conv0 mean: {self.model.model[0].conv.weight.mean().item():.6f}")
                val_results = self.validator()
                print("Val results:", val_results)
                val_map50 = val_results.get('metrics/mAP50(B)', -1.0)
                LOGGER.info(f"Epoch {epoch + 1}/{self.epochs}: 平均损失 = {avg_loss:.4f}, Val mAP50 = {val_map50:.4f}")
                with open(results_csv, "a") as f:
                    f.write(f"{epoch + 1},{avg_loss:.4f},{val_map50:.4f}\n")
                last_ckpt = os.path.join(weights_dir, "last.pt")
                torch.save(self.model.state_dict(), last_ckpt)
                if val_map50 > best_fitness:
                    best_fitness = val_map50
                    best_ckpt = os.path.join(weights_dir, "best.pt")
                    torch.save(self.model.state_dict(), best_ckpt)
        except Exception as e:
            LOGGER.error(f"训练过程中出错: {e}")
            raise
        finally:
            if hasattr(self, 'neck_extractor'):
                self.neck_extractor.remove()

    def extract_neck_features(self, imgs):
        """利用hook机制提取neck特征"""
        self.neck_extractor.clear()
        _ = self.model(imgs)  # 前向传播，hook自动保存特征
        feats = self.neck_extractor.get_features()
        if feats is not None and len(feats) == 3:
            return feats
        else:
            batch_size = imgs.size(0)
            LOGGER.warning("Neck特征数缺失，使用虚拟neck特征！")
            return [
                torch.randn(batch_size, 256, 20, 20).to(self.device),
                torch.randn(batch_size, 512, 10, 10).to(self.device),
                torch.randn(batch_size, 1024, 5, 5).to(self.device)
            ]

    def _extract_neck_channels(self, model):
        # 直接返回 Layer 4, 6, 9 的 out_channels
        seq = model.model
        indices = [4, 6, 9]
        neck_channels = []
        for i in indices:
            layer = seq[i]
            if hasattr(layer, 'cv2') and hasattr(layer.cv2, 'conv'):
                neck_channels.append(layer.cv2.conv.out_channels)
            elif hasattr(layer, 'm') and hasattr(layer, 'cv2'):
                # C2f结构
                neck_channels.append(layer.cv2.conv.out_channels)
            elif hasattr(layer, 'conv'):
                neck_channels.append(layer.conv.out_channels)
            else:
                # fallback简单处理
                neck_channels.append(list(layer.parameters())[0].shape[0])
        return neck_channels