import torch
import torchaudio
import torch.nn as nn

class FastAudioProcessor(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=64, fixed_duration=3.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.fixed_length = int(sample_rate * fixed_duration) # e.g., 48000 samples
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=1024,
            hop_length=512
        )
        
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, audio_path):
        """
        Reads an audio file and returns a Log-Mel Spectrogram tensor.
        """
        waveform, sr = torchaudio.load(audio_path)
        
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Mix to Mono (if stereo)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        current_len = waveform.shape[1]
        
        if current_len > self.fixed_length:
            waveform = waveform[:, :self.fixed_length]
        elif current_len < self.fixed_length:
            pad_amount = self.fixed_length - current_len
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        # Result shape: [channels, n_mels, time_frames]
        mel_spec = self.mel_transform(waveform)
        
        log_mel_spec = self.amplitude_to_db(mel_spec)

        return log_mel_spec

if __name__ == "__main__":
    processor = FastAudioProcessor(sample_rate=16000, fixed_duration=3.0)
    
    print("Audio Processor initialized.")
    print(f"Target Sample Rate: {processor.sample_rate}")
    print(f"Target Input Length: {processor.fixed_length} samples")
    
    spec = processor('./data/Actor_01/01-01-01-01-01-01-01.mp4')
    print(f"Output Spectrogram Shape: {spec.shape}") 
    # Expected: [1, 64, 94] (1 channel, 64 mel bands, ~94 time frames)