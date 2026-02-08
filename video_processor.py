import cv2
import torch
import numpy as np
from torchvision import transforms

class VideoProcessor:
    def __init__(self, num_frames=5, height=224, width=224):
        self.num_frames = num_frames
        self.height = height
        self.width = width
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def process(self, video_path):
        """
        Reads a video and returns a tensor of shape [num_frames, 3, H, W]
        """
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            return torch.zeros((self.num_frames, 3, self.height, self.width))
            
        # 2. Calculate indices for uniform sampling
        # e.g., if video has 100 frames and we want 5: [0, 20, 40, 60, 80]
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            # Jump directly to the frame index (much faster than reading every frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # OpenCV reads in BGR, convert to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Apply transforms (Resize -> Tensor -> Normalize)
                frame_tensor = self.transform(frame)
                frames.append(frame_tensor)
            else:
                # If read fails, pad with a black frame
                frames.append(torch.zeros((3, self.height, self.width)))
                
        cap.release()
        
        # Stack into a single tensor: [num_frames, Channels, Height, Width]
        return torch.stack(frames)

# --- Usage Example ---
if __name__ == "__main__":
    vp = VideoProcessor(num_frames=5)
    tensor = vp.process("./data/Actor_01/01-01-01-01-01-01-01.mp4")
    print(tensor.shape)
    # Expected: torch.Size([5, 3, 224, 224])