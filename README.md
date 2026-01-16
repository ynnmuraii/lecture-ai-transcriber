# Lecture Transcriber

An LLM-based tool for automatic transcription of video lectures and creation of structured notes with GPU acceleration support.

## Features

- **Audio Extraction**: Extract audio from MP4, MKV, and WebM video files
- **Speech Recognition**: Use advanced Whisper models for accurate transcription:
  - `openai/whisper-tiny` - Fast testing and resource-constrained systems
  - `openai/whisper-medium` - Balanced quality and speed (default)
  - `openai/whisper-large-v3` - Maximum quality for powerful hardware
  - `openai/whisper-large-v3-turbo` - Fast processing with high quality
  - `antony66/whisper-large-v3-russian` - Specialized model for Russian language
- **Intelligent Text Processing**: Use Qwen2.5-1.5B-Instruct for:
  - Text preprocessing and filler word removal
  - Intelligent segment merging
  - Technical content identification
  - Text summarization
- **Formula Formatting**: Convert Russian mathematical expressions to proper notation
- **Structured Output**: Generate Markdown notes with timestamps and metadata
- **Resource Management**: Automatic model selection based on system resources
- **Virtual Environment**: Complete dependency isolation

## Requirements

### System Requirements
- Python 3.8 or higher
- FFmpeg (for audio processing)
- 8GB+ RAM recommended (4GB minimum)
- GPU optional but highly recommended for performance

### GPU Requirements (Optional)
- **NVIDIA**: GTX 1060+ or RTX series with 4GB+ VRAM
- **Apple**: M1/M2/M3 with 8GB+ unified memory  
- **AMD**: RX 6000+ series with ROCm support

### Storage Requirements
- Base installation: ~2GB
- Model cache: 5-15GB (depending on selected models)
- Working space: 1-5GB per video file

## Configuration

The system is highly configurable through `config/config.yaml`. Key settings include:

### GPU Settings
```yaml
resources:
  device: "auto"  # Options: "auto", "cpu", "cuda", "mps"
  gpu:
    enabled: true
    memory_fraction: 0.8  # Use 80% of GPU memory
    fallback_to_cpu: true
```

### Model Selection
- **Whisper Models**: Choose based on your hardware:
  - `openai/whisper-tiny` (500MB RAM) - Fast testing and limited resources
  - `openai/whisper-base` (1GB RAM) - Basic quality  
  - `openai/whisper-medium` (2GB RAM) - Balanced quality and speed (recommended)
  - `openai/whisper-large-v3` (8GB RAM) - Maximum quality
  - `openai/whisper-large-v3-turbo` (6GB RAM) - Fast processing with high quality
  - `antony66/whisper-large-v3-russian` (8GB RAM) - Specialized for Russian language

### Text Processing
- **LLM Provider**: Qwen2.5-1.5B-Instruct for intelligent text processing
- **Text Cleaning**: Customizable filler words and cleaning intensity
- **Formula Processing**: Mathematical term mappings for Russian expressions
- **Output Format**: Markdown with timestamps and metadata

## Troubleshooting

### Performance Issues
- Use GPU acceleration when available
- Enable flash attention for compatible models
- Process shorter video segments for large files

## Development

This project follows a modular pipeline architecture with comprehensive testing:

- **Unit Tests**: Test specific functionality and edge cases
- **Property-Based Tests**: Validate universal properties across diverse inputs
- **Integration Tests**: Test end-to-end pipeline functionality

Run tests with:
```bash
pytest tests/
```

## License

MIT License - see LICENSE file for details.