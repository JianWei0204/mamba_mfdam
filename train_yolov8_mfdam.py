"""
YOLOv8与Mamba-MFDAM的训练脚本
"""

from ultralytics import YOLO
from yolov8_mfdam_trainer import YOLOv8MFDAMTrainer
import argparse
import yaml


def create_domain_dataset_config(source_data, target_data, output_path):
    """创建域适应的数据集配置"""
    config = {
        'path': '',  # 根目录
        'train': [source_data, target_data],  # 源域和目标域都用于训练
        'val': source_data,  # 在源域上验证
        'nc': 6,  # 类别数量 (根据您的数据集调整)
        'names': []  # 类别名称
    }

    with open(output_path, 'w') as f:
        yaml.dump(config, f)

    return output_path


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 + Mamba-MFDAM训练脚本')
    # 保持 --model 参数名称
    parser.add_argument('--pretrained', type=str, default='yolov8n.pt', help='YOLOv8模型')
    parser.add_argument('--source-data', type=str, default='/root/autodl-tmp/ultralytics/cfg/datasets/source_dataset.yaml', help='源域数据集')
    parser.add_argument('--target-data', type=str, default='/root/autodl-tmp/ultralytics/cfg/datasets/target_dataset.yaml', help='目标域数据集')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮次')
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--domain-weight', type=float, default=0.1, help='域适应损失权重')
    parser.add_argument('--device', type=str, default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', type=str, default='runs/train', help='项目目录')
    parser.add_argument('--name', type=str, default='yolov8_mfdam', help='实验名称')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--lr0', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='权重衰减')

    args = parser.parse_args()

    # 创建数据集配置
    dataset_config = create_domain_dataset_config(
        args.source_data,
        args.target_data,
        'domain_dataset.yaml'
    )

    # 使用 args.model
    model = YOLO(args.pretrained)

    # 使用MFDAM训练
    results = model.train(
        data=dataset_config,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        trainer=YOLOv8MFDAMTrainer,
        # MFDAM特定参数
        domain_weight=args.domain_weight,
        alpha_schedule='linear',
        max_alpha=1.0
    )

    print(f"训练完成。结果: {results}")

    # 在两个域上测试
    print("\n在源域上评估...")
    source_results = model.val(data=args.source_data)

    print("\n在目标域上评估...")
    target_results = model.val(data=args.target_data)

    print(f"源域 mAP50: {source_results.box.map50:.4f}")
    print(f"目标域 mAP50: {target_results.box.map50:.4f}")


if __name__ == '__main__':
    main()