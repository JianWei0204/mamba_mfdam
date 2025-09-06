from ultralytics import YOLO
from yolov8_mfdam_trainer import YOLOv8MFDAMTrainer
import argparse

def main():
    parser = argparse.ArgumentParser(description='YOLOv8 + Mamba-MFDAM批次级交替训练脚本')
    parser.add_argument('--pretrained', type=str, default='yolov8n.pt')
    parser.add_argument('--source-data', type=str,default='/root/autodl-tmp/ultralytics/cfg/datasets/source_dataset.yaml', help='源域yaml')
    parser.add_argument('--target-data', type=str,default='/root/autodl-tmp/ultralytics/cfg/datasets/target_dataset.yaml', help='目标域yaml')
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

    # 初始化模型
    model = YOLO(args.pretrained)

    # 交替训练
    results = model.train(
        data=args.source_data,  # 只用于初始化类别等
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        trainer=YOLOv8MFDAMTrainer,
        domain_weight=args.domain_weight,
        source_data=args.source_data,
        target_data=args.target_data
    )

    print(f"训练完成。结果: {results}")

    print("\n在源域上评估...")
    source_results = model.val(data=args.source_data)
    print(f"源域 mAP50: {source_results.box.map50:.4f}")

    print("\n在目标域上评估...")
    target_results = model.val(data=args.target_data)
    print(f"目标域 mAP50: {target_results.box.map50:.4f}")

if __name__ == '__main__':
    main()