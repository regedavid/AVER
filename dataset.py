import os
import pandas as pd
import torch
from torch.utils.data import Dataset

from audio_processor import FastAudioProcessor
from video_processor import VideoProcessor
# Define the emotion map based on RAVDESS documentation
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def load_ravdess_data(root_path):
    data = []
    
    # Walk through all directories
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.mp4'):
                # parsing filename: 02-01-06-01-02-01-12.mp4
                parts = file.split('-')
                
                # Safety check: RAVDESS filenames have 7 parts
                if len(parts) != 7:
                    continue
                    
                emotion_code = parts[2]
                actor_code = parts[6].split('.')[0] # Remove .mp4
                
                # Store relative path, label, and actor ID
                data.append({
                    'path': os.path.join(root, file),
                    'emotion': emotion_map.get(emotion_code),
                    'emotion_code': int(emotion_code),
                    'actor': int(actor_code)
                })
                
    return pd.DataFrame(data)


class RAVDESSDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): The dataframe with 'path', 'emotion_code', etc.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        
        # Audio: 3 seconds fixed duration, 16kHz
        self.audio_processor = FastAudioProcessor(
            sample_rate=16000, 
            n_mels=64, 
            fixed_duration=3.0
        )
        
        # Video: 5 frames, 224x224 (Standard ResNet size)
        self.video_processor = VideoProcessor(
            num_frames=5, 
            height=224, 
            width=224
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_path = self.dataframe.iloc[idx]['path']
        
        # 2. Get the label
        # RAVDESS codes are 01 to 08.
        # PyTorch CrossEntropyLoss expects class indices 0 to 7.
        # So we subtract 1.
        emotion_code = self.dataframe.iloc[idx]['emotion_code']
        label = torch.tensor(int(emotion_code) - 1, dtype=torch.long)

        # 3. Process Audio
        # Returns: [1, 64, 94] (Channels, Mel, Time)
        try:
            audio_tensor = self.audio_processor(video_path)
        except Exception as e:
            print(f"Error processing audio for {video_path}: {e}")
            # Return a generic zero tensor if audio fails
            audio_tensor = torch.zeros(1, 64, 94)

        # 4. Process Video
        # Returns: [5, 3, 224, 224] (Frames, Channels, Height, Width)
        try:
            video_tensor = self.video_processor.process(video_path)
        except Exception as e:
            print(f"Error processing video for {video_path}: {e}")
            # Return a generic zero tensor if video fails
            video_tensor = torch.zeros(5, 3, 224, 224)

        # 5. Return as a dictionary
        sample = {
            'audio': audio_tensor,
            'video': video_tensor,
            'label': label,
            'path': video_path # Useful for debugging!
        }

        return sample

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    df = load_ravdess_data("./data")
    print(f"Total samples found: {len(df)}")

    dataset = RAVDESSDataset(dataframe=df, root_dir="")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    for batch in dataloader:
        print("Batch Loaded!")
        print(f"Audio Shape: {batch['audio'].shape}") 
        # Expected: [2, 1, 64, 94] (Batch, Channel, Freq, Time)
        
        print(f"Video Shape: {batch['video'].shape}") 
        # Expected: [2, 5, 3, 224, 224] (Batch, Frames, Channels, H, W)
        
        print(f"Labels: {batch['label']}")
        # Expected: [2, 4] (because 3-1=2, 5-1=4)
        break