# VocalSound
[中文](README.md) | English
## Dataset Introduction
VocalSound is a dataset for human vocal sound recognition. The speakers cover different ages, genders, and nationalities, with a total of more than 21,000 speech files in WAV format. It includes six different types of sounds: laughter, sigh, cough, throat clearing, sneeze, and sniff. The model is required to determine which category each speech file belongs to.

> 🔗 Dataset Homepage Link: [https://huggingface.co/datasets/maoxx241/audio_vocalsound_16k_subset](https://huggingface.co/datasets/maoxx241/audio_vocalsound_16k_subset)

## Dataset Deployment
- The dataset can be obtained from the Hugging Face dataset link 🔗: [https://huggingface.co/datasets/maoxx241/audio_vocalsound_16k_subset](https://huggingface.co/datasets/maoxx241/audio_vocalsound_16k_subset)
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set for dataset tasks). Taking deployment on a Linux server as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
git lfs install
git clone https://huggingface.co/datasets/maoxx241/audio_vocalsound_16k_subset
mv audio_vocalsound_16k_subset vocalsound
mv vocalsound/subset1/* vocalsound/
mv vocalsound/subset2/* vocalsound/
mv vocalsound/subset3/* vocalsound/
mv vocalsound/subset4/* vocalsound/
mv vocalsound/subset5/* vocalsound/
```
- Execute `tree vocalsound/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure matches the one shown below, the dataset has been deployed successfully:
    ```
    vocalsound
    ├── f0003_0_cough.wav
    ├── f0004_0_laughter.wav
    └── f0007_0_sneeze.wav
    # ......
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code File Path |
| --- | --- | --- | --- | --- | --- |
| vocalsound_gen | Generative task for the VocalSound dataset. ⚠️ For this dataset task, the audio path will be directly passed to the service deployment. Ensure that the service deployment supports this input format and has permission to access the audio at the specified path. | Accuracy | 0-shot | List format (contains two types of data: text and audio) | [vocalsound_gen.py](vocalsound_gen.py) |
| vocalsound_gen_base64 | Generative task for the VocalSound dataset. ⚠️ For this dataset task, the audio data will be converted to Base64 format before being passed to the service deployment. Ensure that the service deployment supports this input format. | Accuracy | 0-shot | List format (contains two types of data: text and audio) | [vocalsound_gen_base64.py](vocalsound_gen_base64.py) |