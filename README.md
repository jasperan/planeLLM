# planeLLM
Bite-sized podcasts to learn about anything powrred by the OCI GenAI Service


Requirements: install [OCI CLI](https://docs.oracle.com/en-us/iaas/Content/API/SDKDocs/cliinstall.htm#InstallingCLI__macos_homebrew)

And run the following command with your OCI login information:

```bash
oci setup config
```

In order to authenticate with OCI services and be able to call the OCI GenAI service through the OCI Service Gateway.

In `config.yaml`, you will need to complete these variables (find them in your ))

```yaml
# OCI Configuration
compartment_id: "compartment_ocid"
config_profile: "profile_name"
```

## Usage Options

### Option 1: Unified Pipeline (Recommended)

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

### Option 2: Step-by-Step Execution

If you prefer more control, you can run each step separately. This will create raw lesson content and questions in the resources directory

```bash
python 1_topic_explorer.py --topic "Fort Worth and its history"
# You can replace "Ancient Rome" with any topic you want to learn about
# For example:
python 1_topic_explorer.py --topic "Quantum Physics"
python 1_topic_explorer.py --topic "Machine Learning"
```

### 2. Convert to Podcast Format

Transform the educational content into a conversational podcast format. You can choose between two conversation styles:

```bash
# Two speakers (default): Expert and Student
python 2_lesson_writer.py

# Three speakers: Expert, Student, and Developer
python 2_lesson_writer.py --speakers 3

# Specify a different config file
python 2_lesson_writer.py --config my_config.yaml --speakers 3
```

The three-speaker format adds a developer's perspective to the conversation, including technical insights about implementation, APIs, and DevOps considerations.

All options will generate the same output file (`podcast_transcript.txt`) in the resources directory.

## Use default `suno/bark` model

```bash
python 3_tts.py
# Use `parler` model
python 3_tts.py --model parler
# Specify custom transcript path
python 3_tts.py --transcript ./my_transcript.txt
```

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
