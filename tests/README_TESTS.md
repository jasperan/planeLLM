# planeLLM Tests

This directory contains tests for the planeLLM project components.

## Running Tests

### Running All Tests

To run all tests with coverage report:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests with coverage report
pytest --cov=../ --cov-report=term-missing
```

### Running Specific Tests

To run tests for a specific module:

```bash
# Test the topic explorer
pytest test_topic_explorer.py

# Test the lesson writer
pytest test_lesson_writer.py

# Test the TTS generator
pytest test_tts.py

# Test the podcast controller
pytest test_podcast_controller.py
```

## Parler TTS Audio Generation Test

A specific test for Parler TTS audio generation is available. This test verifies that the Parler TTS model can generate audio from a sample sentence.

### Prerequisites

Make sure you have the Parler TTS package installed:

```bash
pip install git+https://github.com/huggingface/parler-tts.git
```

### Running the Parler Test

You can run the Parler test using the provided script:

```bash
python run_parler_test.py
```

Or using pytest:

```bash
pytest test_parler_audio.py -v
```

### What the Parler Test Checks

The Parler test performs the following checks:

1. **Initialization Test**: Verifies that the TTSGenerator can be initialized with the Parler model
2. **Audio Generation Test**: Tests the `_generate_audio_parler` method with a sample sentence
3. **Podcast Generation Test**: Tests the full podcast generation pipeline with Parler TTS

If Parler TTS is not available on your system, the tests will automatically fall back to using Bark TTS and report that Parler was skipped.

## Test Output

The Parler test will generate a temporary MP3 file to verify that audio generation works correctly. This file is automatically cleaned up after the test completes.

If you want to keep the generated audio for inspection, you can modify the test to disable cleanup by commenting out the relevant lines in the `tearDown` method. 