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
python planellm.py --topic "Ancient Rome"
```

Additional options:
```bash
# Use a different TTS model
python planellm.py --topic "Ancient Rome" --tts-model bark

# Specify a different config file
python planellm.py --topic "Ancient Rome" --config my_config.yaml
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

## Use default `suno/bark` model

```bash
python 3_tts.py
# Use `parler` model
python 3_tts.py --model parler
# Specify custom transcript path
python 3_tts.py --transcript ./my_transcript.txt
```
