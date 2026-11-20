from pathlib import Path
import os.path
import math
import torch
import torchaudio
import folder_paths
import tempfile
import numpy as np
import re
from comfy.utils import ProgressBar
from io import BytesIO
# import soundfile as sf
from transformers import (
    AutoModel, AutoProcessor,
    AutoModelForCausalLM, AutoTokenizer
)
from comfy_api.latest import io


class NoCloseBytesIO(BytesIO):
    def real_close(self):
        # Call this if you genuinely want to close it later
        super().close()

    def close(self):
        # Prevent the underlying BytesIO from closing
        pass


class Audio8Create:
    voice_reg = re.compile(r"\{([^\}]+)\}")
    default_speed_type = "torch-time-stretch"
    speed_types = ["torch-time-stretch", "TDHS"]
    tooltip_speed = "Speed. >1.0 slower. <1.0 faster"
    speed_types_tooltip = "TDHS - Time-domain harmonic scaling. torch-time-stretch - torchaudio.transforms.TimeStretch."  # noqa: E501
    default_sample_text = "Text of sample_audio"

#    optional_inputs = {
#        "sample_audio": ("AUDIO", {
#            "tooltip": "When this is connected, sample is ignored.  Also put the words into sample_text",  # noqa: E501
#        }),
#        "sample_text": ("STRING", {
#            "default": default_sample_text,
#            "multiline": True,
#        }),
# #        "speed_type": (speed_types, {
# #            "default": "torch-time-stretch",
# #            "tooltip": speed_types_tooltip
# #        }),
#    }

    @staticmethod
    def get_txt_file_path(file):
        p = Path(file)
        return os.path.join(os.path.dirname(file), p.stem + ".txt")

    @staticmethod
    def is_voice_name(word):
        return Audio8Create.voice_reg.match(word.strip())

    @staticmethod
    def get_voice_names(chunks):
        voice_names = {}
        for text in chunks:
            match = Audio8Create.is_voice_name(text)
            if match:
                voice_names[match[1]] = True
        return voice_names

    @staticmethod
    def split_text(speech):
        reg1 = r"(?=\{[^\}]+\})"
        return re.split(reg1, speech)

    @staticmethod
    def load_voice(ref_audio, ref_text):
        main_voice = {"ref_audio": ref_audio, "ref_text": ref_text}

        main_voice["ref_audio"] = ref_audio
        main_voice["ref_text"] = ref_text
        return main_voice

    def generate_audio(
        self, voices, model, chunks, seed,
        infer_args={}
    ):
        if seed >= 0:
            torch.manual_seed(seed)
        else:
            torch.random.seed()
        pbar = ProgressBar(len(chunks))

        processor = AutoProcessor.from_pretrained(
            model.config._name_or_path,
            trust_remote_code=True
        )
        input_args = {
            "text": [],
            "reference_audio": [],
            "reference_text": [],
            "return_tensors": "pt",
        }

        if voices is not None:
            for text in chunks:
                match = self.is_voice_name(text)
                if match:
                    voice = match[1]
                else:
                    print("No voice tag found, using main.")
                    voice = "main"
                text = Audio8Create.voice_reg.sub("", text)
                gen_text = text.strip()
                if gen_text == "":
                    print(f"No text for {voice}, skip")
                    continue

                if voice not in voices:
                    print(f"Voice {voice} not found, using main.")
                    voice = "main"

                ref_audio = voices[voice]["ref_audio"]
                ref_text = voices[voice]["ref_text"]

                print(f"Voice: {voice}")
                input_args["reference_audio"].append(ref_audio)
                input_args["reference_text"].append(ref_text)

                print("text:"+text)
                input_args["text"].append(text)
        else:
            input_args["text"] = "".join(chunks)
            del input_args["reference_audio"]
            del input_args["reference_text"]

        inputs = processor(
            **input_args,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {name: value.to(device) for name, value in inputs.items()}

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.8,
                top_p=0.95,
                top_k=50,
                do_sample=True,
                return_dict_in_generate=True,
            )
            waveforms, waveform_lengths = model.decode_audio(output.codes)

        generated_audio_segments = []
        for index in range(0, waveforms.shape[0]):
            audio = waveforms[
                index, : int(waveform_lengths[index])
            ].float().cpu().numpy()
            generated_audio_segments.append(audio)

        final_wave = np.concatenate(generated_audio_segments)
        frame_rate = model.config.codec_sample_rate
        pbar.update(1)

        final_wave = np.concatenate(generated_audio_segments)

        waveform = torch.from_numpy(final_wave).unsqueeze(0)
        audio = {
            "waveform": waveform.unsqueeze(0),
            "sample_rate": frame_rate
            }
        return audio

    def time_shift(self, audio, speed, speed_type):
        if speed == 1:
            return audio
        elif speed_type == "TDHS":
            # windows fatal exception here
            return self.time_shift_audiostretchy(audio, speed)
        elif speed_type == "torch-time-stretch":
            return self.time_shift_torch_ts(audio, speed)
        return audio

    def time_shift_audiostretchy(self, audio, speed):
        from audiostretchy import AudioStretch
        from torchcodec.encoders import AudioEncoder

        rate = audio['sample_rate']
        waveform = audio['waveform']

        audio_stretch = AudioStretch()
        new_waveforms = []
        for channel in range(0, waveform.shape[0]):
            wav_buffer = BytesIO()
            encoder = AudioEncoder(waveform[0][channel], sample_rate=rate)
            encoder.to_file_like(wav_buffer, format="wav")

            wav_buffer.seek(0)  # rewind so it can be read from the start

            audio_stretch.open(file=wav_buffer, format="wav")
            # audiostretchy crashes in windows beyond these
            #  ranges even with double_range=True
            if speed == 0.5:
                speed = 0.5000001  # seg faults at 0.5 or 1
            double_range = False if (speed > 0.5 and speed < 4) else True

            audio_stretch.stretch(
                ratio=speed,
                gap_ratio=1.2,
                upper_freq=333,
                lower_freq=55,
                buffer_ms=25,
                threshold_gap_db=-40,
                double_range=double_range,
                fast_detection=False,
                normal_detection=False,
            )

            out_buffer = NoCloseBytesIO()
            audio_stretch.save(file=out_buffer, format="wav")
            out_buffer.seek(0)  # rewind so it can be read from the start

            waveform, sample_rate = torchaudio.load(out_buffer)
            out_buffer.real_close()

            new_waveforms.append(waveform[0])

        new_waveform = torch.stack(new_waveforms)
        new_waveform = torch.stack([new_waveform])

        return {"waveform": new_waveform, "sample_rate": rate}

    def time_shift_torch_ts(self, audio, speed):
        import torch_time_stretch
        rate = audio['sample_rate']
        waveform = audio['waveform']

        new_waveform = torch_time_stretch.time_stretch(
            waveform,
            torch_time_stretch.Fraction(math.floor(speed*100), 100),
            rate
        )

        return {"waveform": new_waveform, "sample_rate": rate}

    @staticmethod
    def load_voice_from_file(sample):
        input_dir = folder_paths.get_input_directory()
        txt_file = os.path.join(
            input_dir,
            Audio8Create.get_txt_file_path(sample)
            )
        audio_text = ''
        with open(txt_file, 'r', encoding='utf-8') as file:
            audio_text = file.read()
        audio_path = folder_paths.get_annotated_filepath(sample)
        print(f"audio_text {audio_text}")
        return Audio8Create.load_voice(audio_path, audio_text)

    @staticmethod
    def load_voices_from_files(sample, voice_names):
        voices = {}
        p = Path(sample)
        for voice_name in voice_names:
            if voice_name == "main":
                continue
            sample_file = os.path.join(
                os.path.dirname(sample),
                "{stem}.{voice_name}{suffix}".format(
                    stem=p.stem,
                    voice_name=voice_name,
                    suffix=p.suffix
                    )
                )
            print("voice:"+voice_name+","+sample_file+','+sample)
            voices[voice_name] = Audio8Create.load_voice_from_file(sample_file)
        return voices

    @staticmethod
    def remove_wave_file(wave_file_name):
        if wave_file_name is not None:
            try:
                os.unlink(wave_file_name)
            except Exception as e:
                print("Audio8TTS: Cannot remove? "+wave_file_name)
                print(e)

    @staticmethod
    def load_voice_from_input(sample_audio, sample_text):
        wave_file = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False
            )
        wave_file_name = wave_file.name
        wave_file.close()

        hasAudio = False
        for (batch_number, waveform) in enumerate(
            sample_audio["waveform"].cpu()
        ):
            try:
                torchaudio.save(
                    wave_file_name,
                    waveform,
                    sample_audio["sample_rate"],
                    format="WAV"
                    )
            except Exception as e:
                # https://docs.pytorch.org/audio/stable/generated/torchaudio.save.html
                # As of TorchAudio 2.9, this function relies on TorchCodec’s
                # encoding capabilities under the hood.
                # It is provided for convenience, but we do recommend that you
                # port your code to natively use
                # torchcodec’s AudioEncoder class for better performance:
                print("Might be torch 2.9, torchaudio.save did not work")
                print(e)
                # print(e)
                from torchcodec.encoders import AudioEncoder
                encoder = AudioEncoder(
                    waveform,
                    sample_rate=sample_audio["sample_rate"]
                )
                with open(wave_file_name, 'wb') as f:
                    encoder.to_file_like(
                        f,
                        format="wav",  # or "mp3", "ogg", "flac"
                    )

            hasAudio = True
            break
        if not hasAudio:
            raise FileNotFoundError("No audio input")
        r = Audio8Create.load_voice(wave_file_name, sample_text)
        return (r, wave_file_name)

    @staticmethod
    def get_input_audios():
        input_dir = folder_paths.get_input_directory()
        input_dirs = [
                "",
                'audio',
        ]
        files = []
        for dir_short in input_dirs:
            d = os.path.join(input_dir, dir_short)
            if os.path.exists(d):
                dir_files = folder_paths.filter_files_content_types(
                    os.listdir(d), ["audio", "video"]
                    )
                dir_files = [os.path.join(dir_short, s) for s in dir_files]
                files.extend(dir_files)
        filesWithTxt = [""]
        for file in files:
            txtFile = Audio8Create.get_txt_file_path(file)
            if os.path.isfile(os.path.join(input_dir, txtFile)):
                filesWithTxt.append(file)
        filesWithTxt = sorted(filesWithTxt)
        return filesWithTxt

    # Load from either sample text or sample_audio
    @staticmethod
    def get_chunks_voices(sample_audio, sample_text, sample, speech):
        voices = {}
        if sample_audio is not None:
            if sample_text == Audio8Create.default_sample_text:
                raise Exception(
                    "Must change sample_text to what was said in the audio input."  # noqa: E501
                )
            (
                main_voice, wave_file_name
            ) = Audio8Create.load_voice_from_input(
                    sample_audio, sample_text
                )
            chunks = Audio8Create.split_text(speech)
        else:
            if sample_audio is not None:
                raise Exception(
                    "Should only link sample_audio or sample."
                )

            main_voice = Audio8Create.load_voice_from_file(sample)
            chunks = Audio8Create.split_text(speech)
            voice_names = Audio8Create.get_voice_names(chunks)
            voices = Audio8Create.load_voices_from_files(
                sample, voice_names
            )
        voices['main'] = main_voice
        return (chunks, voices)


class Audio8TTSFromModel(io.ComfyNode):
    def __init__(self):
        pass

    @classmethod
    def define_schema(cls) -> io.Schema:
        filesWithTxt = Audio8Create.get_input_audios()
        return io.Schema(
            node_id="Audio8TTSFromModel",
            display_name="Audio8 Text to Speech",
            category="Audio",
            inputs=[
                io.Model.Input(
                    "model",
                ),
                io.String.Input(
                    "speech",
                    multiline=True,
                    default="This is what I want to say",
                ),
                io.Int.Input(
                    "seed",
                ),
                io.Combo.Input(
                    "sample",
                    options=filesWithTxt,
                ),
                io.Audio.Input(
                    "sample_audio",
                    tooltip="When this is connected, sample is ignored.  Also put the words into sample_text",  # noqa: E501
                    optional=True,
                ),
                io.String.Input(
                    "sample_text",
                    multiline=True,
                    default=Audio8Create.default_sample_text,
                    optional=True,
                ),
                io.Float.Input(
                    "speed",
                    default=1.0,
                    min=0.0,
                    step=0.01,
                    display_mode=io.NumberDisplay.number,
                    tooltip=Audio8Create.tooltip_speed,
                ),
                io.Combo.Input(
                    "speed_type",
                    default="TDHS",
                    options=Audio8Create.speed_types,
                    tooltip=Audio8Create.speed_types_tooltip
                ),
            ],
            outputs=[
                io.Audio.Output(),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        speech,
        seed,
        sample=None,
        sample_audio=None,
        sample_text="",
        speed=1.0,
        speed_type="TDHS",
    ) -> io.NodeOutput:
        audio8Create = Audio8Create()

        infer_args = {}
        if (sample is None or sample == "") and sample_audio is None:
            audio = audio8Create.generate_audio(
                None,
                model,
                [speech], seed,
                infer_args=infer_args,
            )
        else:
            (chunks, voices) = Audio8Create.get_chunks_voices(
                sample_audio, sample_text, sample, speech
            )

            audio = audio8Create.generate_audio(
                voices,
                model,
                chunks, seed,
                infer_args=infer_args,
            )

        audio = audio8Create.time_shift(
            audio, speed, speed_type,
            )

        return (audio, )


class Audio8TTSLoadModel(io.ComfyNode):
    def __init__(self):
        pass

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Audio8TTSLoadModel",
            display_name="Load Audio8-TTS Text to Speech model",
            category="Audio",
            inputs=[
                io.String.Input(
                    "model_id",
                    multiline=False,
                    default="Edge0/Audio8-TTS-Realistic-Air",
                    tooltip="Model id on huggingface.  Edge0/Audio8-TTS-Realistic-Air",  # noqa: E501
                )
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model_id) -> io.NodeOutput:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=dtype,
        ).eval().to(device)

        return io.NodeOutput(model)


class Audio8ASRFromModel(io.ComfyNode):
    def __init__(self):
        pass

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Audio8ASRFromModel",
            display_name="Audio8 Speech to Text",
            category="Audio",
            inputs=[
                io.Model.Input(
                    "model",
                ),
                io.Audio.Input(
                    "speech_audio",
                ),
            ],
            outputs=[
                io.String.Output(),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        speech_audio
    ) -> io.NodeOutput:
        model_path = model.config._name_or_path

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        def build_bad_words_ids(tokenizer):
            eos_ids = tokenizer.eos_token_id
            keep_ids = (
                {eos_ids}
                if isinstance(eos_ids, int)
                else set(eos_ids or [])
            )
            bad_ids = set(tokenizer.all_special_ids) - keep_ids
            bad_ids.update(
                token_id
                for token, token_id in tokenizer.get_added_vocab().items()
                if (
                    token.startswith("<") and
                    token.endswith(">") and
                    token_id not in keep_ids
                )
            )
            return [[token_id] for token_id in sorted(bad_ids)]
        orig_sample_rate = speech_audio["sample_rate"]
        sample_rate = 16000

        waveform = speech_audio["waveform"]
        if sample_rate != orig_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=sample_rate)
            waveform = resampler(waveform)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "array": waveform},
                    {"type": "text", "text": "Please transcribe this audio."},
                ],
            }
        ]


        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            sampling_rate=sample_rate,
            audio_padding="longest",
            text_kwargs={"padding": "longest"},
            audio_max_length=30 * sample_rate,
        )
        inputs = inputs.to(device)
        if "audios" in inputs:
            inputs["audios"] = inputs["audios"].to(dtype=torch_dtype)

        bad_words_ids = build_bad_words_ids(tokenizer)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                bad_words_ids=bad_words_ids,
            )

        decoded_outputs = tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        print(decoded_outputs)

        return ("\n".join(decoded_outputs), )


# Can't run demo fo other  models.
# Edge0/Audio8-ASR-0.1B:  language_model.lm_head.weight | MISSING | Notes: - MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
# Edge0/Audio8-ASR-Infinite:  RuntimeError: Audio8 ASR Infinite checkpoint does not exactly match the current weight format. Convert it before loading. incompatible={'missing_keys': {'semantic_vad_heads.3.bias', 'semantic_vad_heads.2.weight', 'semantic_vad_heads.1.bias', 'semantic_vad_heads.0.bias', 'semantic_vad_heads.2.bias', 'semantic_vad_heads.3.weight', 'semantic_vad_heads.0.weight', 'semantic_vad_heads.1.weight'}}

class Audio8ASRLoadModel(io.ComfyNode):
    def __init__(self):
        pass

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Audio8ASRLoadModel",
            display_name="Load Audio8-ASR Speech to text model",
            category="Audio",
            inputs=[
                io.String.Input(
                    "model_id",
                    multiline=False,
                    default="Edge0/ARK-ASR-3B",
                    tooltip="Model id on huggingface.  Edge0/ARK-ASR-3B, Edge0/ARK-ASR-0.6B",  # noqa: E501
                )
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model_id) -> io.NodeOutput:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",
        ).to(device)
        model.eval()

        return io.NodeOutput(model)
