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