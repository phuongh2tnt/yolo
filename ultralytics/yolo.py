from ultralytics import YOLO
import os
import torch

def check_gpu_availability():
    """
    Kiểm tra và hiển thị thông tin GPU
    """
    print("=== Kiểm tra GPU ===")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✓ CUDA có sẵn! Số GPU: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Hiển thị GPU hiện tại
        current_device = torch.cuda.current_device()
        print(f"GPU hiện tại: {current_device}")
        
        return True, list(range(gpu_count))
    else:
        print("✗ CUDA không có sẵn, sẽ sử dụng CPU")
        return False, []

def main():
    """
    Hàm chính để training YOLO model
    """
    # Kiểm tra GPU
    cuda_available, available_gpus = check_gpu_availability()
    
    # Kiểm tra xem file model có tồn tại không
    model_path = "yolo11n.pt"
    if not os.path.exists(model_path):
        print(f"Cảnh báo: File model {model_path} không tồn tại!")
        print("Model sẽ được tải xuống tự động...")
    
    # Khởi tạo model YOLO
    print("Đang khởi tạo YOLO model...")
    model = YOLO(model_path)
    
    # Cấu hình training parameters
    dataset_config = "/content/yolo/ultralytics/cfg/datasets/databoxtugan.yaml"
    
    # Kiểm tra file dataset config
    if not os.path.exists(dataset_config):
        print(f"Lỗi: File cấu hình dataset {dataset_config} không tồn tại!")
        return
    
    # Cấu hình device và batch size dựa trên GPU
    if cuda_available:
        if len(available_gpus) == 1:
            device = 0  # Sử dụng GPU đầu tiên
            batch_size = 16  # Batch size cho single GPU
        else:
            device = available_gpus  # Sử dụng tất cả GPU
            batch_size = 16 * len(available_gpus)  # Tăng batch size cho multi-GPU
    else:
        device = 'cpu'
        batch_size = 8  # Batch size nhỏ hơn cho CPU
    
    print("Bắt đầu quá trình training...")
    print(f"Dataset: {dataset_config}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print("Epochs: 500")
    print("Image size: 640")
    
    try:
        # Training model với GPU optimization
        train_results = model.train(
            data=dataset_config,  # Path to dataset configuration file
            epochs=500,  # Number of training epochs
            imgsz=640,  # Image size for training
            device=device,  # Tự động sử dụng GPU/CPU tối ưu
            batch=batch_size,  # Batch size tự động điều chỉnh
            workers=8,  # Số worker threads cho data loading
            patience=50,  # Early stopping patience
            save_period=25,  # Save checkpoint every 25 epochs
            amp=True,  # Automatic Mixed Precision (tăng tốc training trên GPU)
            cache=True,  # Cache images trong RAM để tăng tốc
            close_mosaic=10,  # Tắt mosaic augmentation trong 10 epochs cuối
            # Các tham số optimization khác
            optimizer='AdamW',  # Optimizer tối ưu
            lr0=0.01,  # Learning rate ban đầu
            lrf=0.01,  # Final learning rate (lr0 * lrf)
            momentum=0.937,  # SGD momentum/Adam beta1
            weight_decay=0.0005,  # Optimizer weight decay
            warmup_epochs=3.0,  # Warmup epochs
            warmup_momentum=0.8,  # Warmup initial momentum
            box=7.5,  # Box loss gain
            cls=0.5,  # Class loss gain
            dfl=1.5,  # DFL loss gain
        )
        
        print("Training hoàn thành thành công!")
        print(f"Kết quả được lưu tại: {train_results.save_dir}")
        
        # In ra một số thông tin quan trọng
        if hasattr(train_results, 'maps'):
            print(f"mAP50: {train_results.maps}")
        
        return train_results
        
    except Exception as e:
        print(f"Lỗi trong quá trình training: {str(e)}")
        return None

if __name__ == "__main__":
    print("=== YOLO Training Script ===")
    results = main()
    
    if results:
        print("Training script kết thúc thành công!")
    else:
        print("Training script gặp lỗi!")
