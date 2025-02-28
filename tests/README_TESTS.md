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
# Run all Parler tests
python run_parler_test.py

# Run only the voice selection test
python run_parler_test.py --voice-test

# Run specific tests
python run_parler_test.py --init        # Test initialization only
python run_parler_test.py --generate    # Test audio generation only
python run_parler_test.py --podcast     # Test podcast generation only
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
4. **Voice Selection Test**: Tests the ability to select different voice styles for each speaker

### Voice Selection Feature

The Parler TTS implementation now supports selecting different voice styles for each speaker. The available voice styles are:

**Male Voices:**
- `male_clear`: Jon's voice - clear and professional with a moderate pace
- `male_expressive`: Gary's voice - expressive and animated with varied intonation
- `male_deep`: Bruce's voice - deep and authoritative with a measured pace
- `male_casual`: Rick's voice - casual and conversational with a natural flow

**Female Voices:**
- `female_clear`: Laura's voice - clear and professional with a moderate pace
- `female_expressive`: Jenna's voice - expressive and animated with varied intonation
- `female_warm`: Rose's voice - warm and engaging with a pleasant tone
- `female_casual`: Lea's voice - casual and conversational with a natural flow

You can set a voice style for a speaker using the `set_voice_type` method:

```python
tts = TTSGenerator(model_type="parler")
tts.set_voice_type("Speaker 1", "male_expressive")
tts.set_voice_type("Speaker 2", "female_warm")
```

You can also set a custom voice description using the `set_voice_description` method:

```python
tts = TTSGenerator(model_type="parler")
tts.set_voice_description("Speaker 1", "Jon's voice is clear and professional with very clear audio.")
```

## Test Output

The Parler test will generate a temporary MP3 file to verify that audio generation works correctly. This file is automatically cleaned up after the test completes.

If you want to keep the generated audio for inspection, you can modify the test to disable cleanup by commenting out the relevant lines in the `tearDown` method. 