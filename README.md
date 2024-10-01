### README: Noise Cancellation Using Generative AI

---

# Noise Cancellation Using Generative AI

This project aims to develop a noise cancellation system using a Generative Adversarial Network (GAN). The goal is to take a noisy audio file as input, generate the inverse waveform, and combine the two signals to achieve noise cancellation (i.e., producing silence). The project is implemented in Python, using libraries like `PyTorch`, `Librosa`, and `Matplotlib`, and includes the option for API integration (e.g., OpenAI or Google AI) for advanced audio processing.

---

## Project Overview

- **Input**: Noisy audio waveform (`.wav` format).
- **Output**: Predicted inverse audio waveform that cancels the noise.
- **Model**: Generative Adversarial Network (GAN) consisting of a Generator and Discriminator.
- **Libraries**: `PyTorch`, `Librosa`, `Matplotlib`, `SoundFile`, `NumPy`, `TensorFlow`.
- **Visualization**: Plot audio waveforms and spectrograms to illustrate noise cancellation results.

---

## Features

- **Noise Cancellation**: Generates an inverse waveform for any noisy audio signal.
- **GAN Architecture**: 
  - Generator: Learns to generate the inverse waveform.
  - Discriminator: Helps improve the quality of the inverse waveform by distinguishing between real and generated samples.
- **Audio Processing**: Uses `Librosa` for loading audio files and converting waveforms to spectrograms for model input.
- **Training and Inference**: The notebook provides a comprehensive training loop and inference mechanism.
- **External API Integration**: Supports external APIs like OpenAI or Google AI to offload audio processing to cloud-based services.
- **Visualization**: Audio waveforms and spectrograms are plotted using `matplotlib` for easy inspection of the results.

---

## Getting Started

### Prerequisites

Ensure you have the following software and libraries installed:

- Python 3.x
- Jupyter Notebook
- The following Python packages (can be installed via `pip`):
  - `numpy`
  - `scipy`
  - `librosa`
  - `soundfile`
  - `matplotlib`
  - `torch`
  - `torchaudio`
  - `tensorflow` (optional if you prefer PyTorch)
  
### Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your_username/noise-cancellation-ai.git
   cd noise-cancellation-ai
   ```

2. Install the required dependencies:
   ```bash
   pip install numpy scipy librosa soundfile matplotlib torch torchaudio tensorflow
   ```

3. Open the Jupyter notebook:
   ```bash
   jupyter notebook noise_cancellation.ipynb
   ```

---

## Usage

### Step-by-Step Guide

1. **Load Audio**: Use `librosa` to load a noisy audio file in `.wav` format. You can replace `path_to_wav_file.wav` with your own file.
   
2. **Convert to Spectrogram**: Convert the waveform to a spectrogram using the Short-Time Fourier Transform (STFT). This makes the audio data suitable for the neural network.

3. **GAN Training**: Train the GAN model to predict the inverse waveform. The generator attempts to create the inverse waveform while the discriminator ensures that the generated waveform is close to the real inverse.

4. **Inference**: After training, use the generator to create the inverse waveform for any new input audio and visualize the noise cancellation by plotting the original and combined waveforms.

5. **Optional API Integration**: Optionally, you can integrate external APIs such as OpenAI or Google AI for additional audio processing services.

---

### Sample Code Snippets

#### Load and Visualize Audio
```python
audio, sr = load_audio('path_to_wav_file.wav')
plot_waveform(audio, sr)
```

#### Convert to Spectrogram
```python
spectrogram = audio_to_spectrogram(audio, sr)
plot_spectrogram(spectrogram, sr)
```

#### GAN Training
```python
train(generator, discriminator, data_loader, optimizer_G, optimizer_D, criterion, num_epochs=100)
```

#### Inference and Visualization
```python
inverse_waveform = spectrogram_to_audio(generator(spectrogram), sr)
combined_waveform = audio + inverse_waveform
plot_waveform(combined_waveform, sr)
```

---

## External API Integration (Optional)

If you'd like to use external APIs for noise cancellation, you can integrate services such as OpenAI or Google AI. Here's an example of how to send the audio data to an API:

```python
import openai

def call_openai_audio_model(audio):
    response = openai.Audio.create(audio_file=audio, ...)
    return response['output_audio']

# Example usage
output_audio = call_openai_audio_model('path_to_noisy_audio.wav')
```

---

## Visualization

The notebook provides visualizations of:
- **Waveforms**: Before and after noise cancellation.
- **Spectrograms**: To illustrate how the frequencies change across time.

---

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Future Work

- **Improve Model Architecture**: Experiment with other generative models like VAEs (Variational Autoencoders) for noise cancellation.
- **Real-Time Processing**: Extend the model to handle real-time audio streaming for live noise cancellation.
- **API Integration**: Further develop the integration with external APIs for more complex audio processing scenarios.
- **Model Optimization**: Fine-tune the model for better performance with different types of noise (e.g., white noise, background chatter).

---

## References

- [Librosa Documentation](https://librosa.org/doc/latest/index.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [GANs in PyTorch](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [OpenAI API Documentation](https://beta.openai.com/docs/)

---

## Contact

For any questions or issues, please feel free to open an issue on GitHub or contact me at hdelahitte@gmail.com

---

### Example Visualizations

Original Waveform:

![Original Waveform](images/original_waveform.png)

Combined Waveform (After Noise Cancellation):

![Combined Waveform](images/combined_waveform.png)

---

### Acknowledgements

Special thanks to all the open-source libraries and frameworks that made this project possible!

---

**Happy Coding! 🎧🚀**

