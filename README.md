# Wav2Vec2 for Noise Cancellation (Waveform Inversion)

This project demonstrates how to fine-tune the **Wav2Vec2** model from Hugging Face for the task of **noise cancellation through waveform inversion**. The model learns to generate the inverse (phase-shifted) version of an input signal, so that when the original input and the generated inverse are combined, the result is **silence** due to destructive interference.

## Goal
The goal of this project is to train a machine learning model that can:
- Take a noisy input signal (e.g., `.wav` file).
- Generate an **inverse waveform** that, when added to the original signal, produces a flat, silent output.

This is useful for applications where you want to cancel out unwanted noise or signals using phase-shifted waveforms.

## Model: Wav2Vec2
- **Pre-trained Model**: Wav2Vec2 (`facebook/wav2vec2-base-960h`), which was originally designed for speech recognition tasks.
- **Fine-tuning Task**: Instead of speech recognition, the model is fine-tuned to generate the inverse of the input audio signal (180° phase shift).

## Key Features
- **Waveform Cancellation**: The model learns to generate the inverse waveform to cancel the original input.
- **Silence as Output**: When the predicted inverse signal is added to the original, the output is silence (destructive interference).
- **Visualizations**: Plots are provided to visualize the input signal, the inverse waveform, and the combined result.
- **Custom Dataset**: You can upload your own noisy audio dataset for training.
  
## Project Structure
- **Colab Notebook**: The core implementation is written as a Jupyter Notebook designed to run on Google Colab.
- **Fine-Tuning**: Wav2Vec2 is fine-tuned on a custom dataset where the model learns to invert the input signal.
- **Inference**: After fine-tuning, the model can generate inverse waveforms for any new noisy input.

## Prerequisites

To run this project, you'll need:
- Python 3.x
- Google Colab (or a local Jupyter environment)
- Libraries:
  - Hugging Face's `transformers`
  - `datasets`
  - `librosa` for audio processing
  - `torch` for deep learning

In Google Colab, the required libraries will be installed automatically by the notebook.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/wav2vec2-noise-cancellation.git
cd wav2vec2-noise-cancellation

You can also copy the code from the provided Colab notebook and run it in your own environment.

## Running the Notebook in Google Colab

1. Open the Colab notebook (`wav2vec2_noise_cancellation.ipynb`) in Google Colab.
2. Install the required libraries using the following command:
   ```bash
   !pip install transformers datasets librosa soundfile torch torchaudio
   ```
3. Upload your noisy `.wav` audio file when prompted.
4. The notebook will fine-tune the Wav2Vec2 model to generate the inverse waveform for your noisy input.
5. After training, you can visualize the results and see how the input and inverse signals combine to form silence.

## Usage

### 1. Upload Noisy Audio
Upload your noisy `.wav` audio file when prompted in the notebook. The file will be used as the input signal for training and inference.

### 2. Fine-Tuning
The model will be fine-tuned to predict the inverse waveform (180-degree phase-shifted) of the input signal. This inverse signal is generated to cancel out the input signal.

### 3. Inference
Once fine-tuning is complete, you can run inference on any noisy input signal. The output will be the inverse waveform, which when combined with the original, will result in silence.

### 4. Visualization
The notebook provides visualizations to help you understand the process:
- **Input Signal**: The original noisy signal.
- **Inverse Signal**: The model-generated inverse waveform.
- **Combined Signal**: The combination of input and inverse signals, which should approach silence.

## Training Process

The training process involves fine-tuning Wav2Vec2 using the following steps:
1. **Preprocessing**: Tokenize the noisy input and generate the inverse waveform as the target.
2. **Model Training**: The Wav2Vec2 model is trained to predict the inverse waveform that cancels out the input.
3. **Evaluation**: After training, the model's ability to produce accurate inverse waveforms is evaluated.

## Results

- **Input Audio**: A noisy input audio signal is provided.
- **Inverse Audio**: The model generates an inverse waveform, which is 180° out of phase with the input.
- **Combined Audio**: When the input and inverse audio signals are added together, the result is a silent waveform.

## Future Work

- **Real-time Processing**: Extend the model to handle real-time noise cancellation applications.
- **Model Optimization**: Improve the model architecture for faster and more efficient waveform inversion.
- **Diverse Noise Types**: Fine-tune the model on a variety of noise types to generalize across different audio environments.

## Contributing

Contributions are welcome! If you'd like to contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Submit a Pull Request with a detailed description of the changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Hugging Face's `transformers` library for making Wav2Vec2 easily accessible.
- Google Colab for providing free GPU resources for training models.
- Librosa for audio processing utilities.

### Key Points in the README:

- **Project Overview**: Briefly explains the goal of waveform cancellation using Wav2Vec2.
- **Installation Instructions**: Steps for cloning the repo and running the notebook in Google Colab.
- **Detailed Usage**: Steps for using the notebook, including uploading audio, fine-tuning the model, running inference, and visualizing the results.
- **Future Work**: Suggestions for future improvements, including real-time noise cancellation and support for diverse noise types.

