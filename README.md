
ComfyUI nodes for [https://github.com/Edge0-AI/Audio8_TTS/](Audio8)

* Text to speech (Edge0/ARK-ASR-3B, Edge0/ARK-ASR-0.6B)
* Speech to text (Edge0/Audio8-TTS-Realistic-Air)

[Example workflows](example_workflows/) or you can get them from the templates section in ComfyUI



| Model | VRAM |
| :--- | ---: |
| Audio8-TTS-Realistic-Air (uses more memory with multiple voices) | 2GB+ |
| ARK-ASR-0.6B | 3GB |
| ARK-ASR-3B | 8GB |


### Multi voices output

Put your sample voice files into the `input` folder like...
```
LanguageTest.flac
LanguageTest.txt
LanguageTest.French.flac
LanguageTest.French.txt
LanguageTest.German.flac
LanguageTest.German.txt
```

Then you can use prompts for different voices...
```
{main} Could you tell us where the best place for fish & chips is around here?
{German} You must try the little place down by the harbour, just past the ferry wharf, because the fish is caught fresh that morning.
{French} Perfect, let’s grab lunch there first!
```

The `main` voice refers to the one with no name `LanguageTest.flac`.

Example audio output
https://github.com/user-attachments/assets/d9da26b6-000d-4a8c-9ca0-97652fd2ce20




### Security

Note: The models use trust\_remote\_code=True, which means whomever controls the huggingface repo can do things on your machinhe.  Same with this extension.  If you'd like to use .safetensors, try this extension for Audio8-TTS-Realistic-0.6b TTS: https://github.com/Pepehoschi/ComfyUI-Audio8TTS/
