# planeLLM
Bite-sized podcasts to learn about anything powered by the OCI GenAI Service

## Setup

Requirements: install [OCI CLI](https://docs.oracle.com/en-us/iaas/Content/API/SDKDocs/cliinstall.htm#InstallingCLI__macos_homebrew)

And run the following command with your OCI login information:

```bash
oci setup config
```

In order to authenticate with OCI services and be able to call the OCI GenAI service through the OCI Service Gateway.

In `config.yaml`, you will need to complete these variables:

```yaml
# OCI Configuration
compartment_id: "compartment_ocid"
config_profile: "profile_name"
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage Options

### Option 1: Gradio Web Interface (Recommended)

Run the Gradio web interface for an interactive experience:

```bash
python gradio_app.py
```

The interface provides three tabs:
1. **Topic Explorer**: Generate educational content about any topic
2. **Lesson Writer**: Convert content into podcast transcript format
3. **TTS Generator**: Generate audio from the transcript

Features:
- User-friendly interface with progress indicators
- File selection for content and transcripts
- Choice of TTS models (Bark or Parler)
- Automatic file saving with timestamps

### Option 2: Unified Pipeline

Run the entire pipeline with a single command:
```bash
python podcast_controller.py --topic "Ancient Rome"
```

Additional options:
```bash
# Use a different TTS model
python podcast_controller.py --topic "Ancient Rome" --tts-model bark

# Specify a different config file
python podcast_controller.py --topic "Ancient Rome" --config my_config.yaml
```

### Option 3: Using as a Package

You can also use planeLLM as a Python package:

```python
from topic_explorer import TopicExplorer
from lesson_writer import PodcastWriter
from tts_generator import TTSGenerator

# Generate educational content
explorer = TopicExplorer()
content = explorer.generate_full_content("Ancient Rome")

# Create podcast transcript
writer = PodcastWriter()
transcript = writer.create_podcast_transcript(content)

# Generate audio
generator = TTSGenerator(model_type="bark")
generator.generate_podcast("./resources/podcast_transcript_TIMESTAMP.txt")
```

## Core Components

The project consists of three main modules:

1. **Topic Explorer** (`topic_explorer.py`): Generates educational content about a topic using OCI GenAI service
   - Generates relevant questions about the topic
   - Creates detailed answers for each question
   - Saves questions and content to separate files

2. **Lesson Writer** (`lesson_writer.py`): Transforms educational content into podcast format
   - Converts raw content into a conversational format
   - Supports 2 or 3 speakers
   - Creates natural dialogue between expert(s) and student

3. **TTS Generator** (`tts_generator.py`): Converts podcast transcripts to audio
   - Supports multiple TTS models (Bark and Parler)
   - Handles speaker separation
   - Generates high-quality audio output

## Testing

The project includes comprehensive unit tests for all modules. To run the tests:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests with coverage report
pytest tests/ --cov=./ --cov-report=term-missing

# Run tests for a specific module
pytest tests/test_topic_explorer.py
pytest tests/test_lesson_writer.py
pytest tests/test_tts.py
```

The tests cover:
- Content generation functionality
- Podcast transcript creation
- Audio generation
- Error handling
- Execution time tracking

Each module has its own test file in the `tests/` directory:
- `test_topic_explorer.py`: Tests for educational content generation
- `test_lesson_writer.py`: Tests for podcast script creation
- `test_tts.py`: Tests for audio generation
