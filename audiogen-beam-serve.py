import json
import math
import random
import time
import tempfile
import base64
from fastapi.responses import FileResponse
import numpy as np
from openai import OpenAI
import torch
from audiocraft.data.audio import audio_write
from audiocraft.data.audio_utils import normalize_audio
from audiocraft.models import AudioGen
from beam import QueueDepthAutoscaler, Volume, endpoint, Image
import pathlib
from multiprocessing.dummy import Pool
from urllib.request import urlretrieve
from fastapi import Response


DEFAULT_DURATION = 10.0
MIN_ACCEPTABLE_AMPLITUDE = 0.05 # below ~0.025 is imperceptible, ~0.03-0.1 is very quiet to quiet, above 0.1 is easily perceptible (in my opinion using the touch pro controllers P=1)
MIN_ACCEPTABLE_995TH_AMPLITUDE = 0.02 # at least 0.5% of samples should be above this amplitude (50ms for 10 seconds total duration)
N_AT_ONCE = 5 # no increase in generation time for 5x, apparently
SORTED_TOP_N = 3 # if sorted, return top N wavs

MODEL_HOST_PATH = "https://file_host/audiogen-models/"
CACHE_PATH = "./tag_out_weights"

# can only fit two models in memory on a T4
AUDIO_MODEL_IDS = [
	"HFaudiogen-medium_db34c85a"
]
HAPTIC_MODEL_IDS = [
	# "51eabea7_12c0dcd3",
	# "51eabea7_1f457268"
	"51eabea7_d684b3a7"
]
MODEL_IDS = HAPTIC_MODEL_IDS + AUDIO_MODEL_IDS
def init_models():
	model_dict: dict[str, AudioGen] = {}
	for model_id in MODEL_IDS:
		model_dict[model_id] = init_model(model_id)

	oai_client = OpenAI()

	return model_dict, oai_client

def init_model(model_id):
	model_path = f"{CACHE_PATH}/{model_id}/model/"
	state_dict_path = model_path + 'state_dict.bin'
	compression_state_dict_path = model_path + 'compression_state_dict.bin'
	model_url_base = f"{MODEL_HOST_PATH}/{model_id}/model/"

	pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
	sdp = pathlib.Path(state_dict_path)
	csdp = pathlib.Path(compression_state_dict_path)

	if sdp.exists() and csdp.exists() and sdp.stat().st_size > 3678400000 and csdp.stat().st_size > 235700000:
		print(f"Model files for {model_id} already exist, skipping download.")
	else:
		print("Downloading model files...")
		# Download state_dict.bin and compression_state_dict.bin from MODEL_URL_BASE in parallel to the model directory
		urls = [
			model_url_base + "state_dict.bin",
			model_url_base + "compression_state_dict.bin"
		]
		with Pool(2) as p:
			p.map(lambda url: urlretrieve(url, model_path + url.split("/")[-1]), urls)
		print("Download complete.")


	# Load the model
	print("Loading model...")
	model = AudioGen.get_pretrained(model_path)
	model.set_generation_params(duration=DEFAULT_DURATION)
	return model


def get_prompt_variants(oai_client: OpenAI, prompt: str, n_variants: int = 5) -> list[str]:
	completion = oai_client.beta.chat.completions.parse(
		model="gpt-4o-mini-2024-07-18",
		messages=[
			# {"role": "system", "content": f"Generate {n_variants} unique caption variants for a generative haptic model." +" Use clear and natural 3rd person language. Avoid creative flourishes and stick to straightforward descriptions of the haptic sensation. Avoid repetitive language and focus on variety. Prioritize brevity and clarity, avoiding unnecessary words and pronouns."},
			{"role": "system", "content": f"Generate {n_variants} unique caption variants based on an input prompt for a generative model. Use clear and natural 3rd person language. Avoid creative flourishes and stick to straightforward captions. Avoid repetitive language and focus creating a variety that covers the spectrum of possible generations."},
			{"role": "user", "content": prompt }
		],
		response_format={
			"type": "json_schema",
			"json_schema": {
				"name": "caption_variants",
				"strict": True,
				"schema": {
					"type": "object",
					"properties": {
						"caption_variants": {
							"type": "array",
							"items": {
								"type": "string"
							},
							# "minItems" : n_variants,
							# "maxItems" : n_variants,
							# "uniqueItems" : True
						}
					},
					"required": ["caption_variants"],
					"additionalProperties": False,
				}
			}
		}
	)
	return json.loads(completion.choices[0].message.content)["caption_variants"]


@endpoint(
	cpu=1,
	memory="16Gi",
	gpu="T4",
	keep_warm_seconds=3000, # 50 minutes
	# autoscaler=QueueDepthAutoscaler(max_containers=2, tasks_per_container=1), # does not work as intended, random errors, higher latency even after warmup (not sure if tasks_per_container=0 would help)
	volumes=[Volume(name="tag_out_weights", mount_path=CACHE_PATH)],
	image=Image(
		python_version="python3.9",
		python_packages="requirements.beamcloud.txt", # no comments allowed in this
		commands=["apt-get update -y && apt-get install ffmpeg -y"],
	),
	secrets=["OPENAI_API_KEY"],
	on_start=init_models
)
def inference(context, prompt,
	model_name: str = "51eabea7_12c0dcd3",
	n_at_once: int = N_AT_ONCE,
	resp_type: str = "single",
	sorted_top_n: int = SORTED_TOP_N,
	duration: float = DEFAULT_DURATION,
	use_sampling: bool = True,
	top_p: float = 0.0,
	top_k: int = 250,
	cfg_coef: float = 3.0,
	temperature: float = 1.0,
	create_variants: bool = False,
	normalize_output: bool = False,
):
	model_dict: dict[str, AudioGen]
	oai_client: OpenAI
	model_dict, oai_client = context.on_start_value
	assert isinstance(model_dict, dict), f"model_dict is not a dict, but a {type(model_dict)}"
	assert isinstance(oai_client, OpenAI), f"oai_client is not an OpenAI, but a {type(oai_client)}"

	if model_name not in model_dict:
		raise ValueError(f"Model {model_name} not found in model_dict")
	model: AudioGen = model_dict[model_name]

	model.set_generation_params(duration=duration, use_sampling=use_sampling, top_p=top_p, top_k=top_k, temperature=temperature, cfg_coef=cfg_coef)

	model_is_haptic = model_name in HAPTIC_MODEL_IDS

	if create_variants:
		get_variants_start_time = time.time()
		prompt_list = [prompt] + get_prompt_variants(oai_client, prompt, n_variants=n_at_once - 1)
		# print(f"Took {time.time() - get_variants_start_time} seconds for {prompt_list}")
		print(f"Took {time.time() - get_variants_start_time} seconds for variants")
	else:
		prompt_list = [prompt] * n_at_once

	print(f"gpus avail: {torch.cuda.device_count()}, {torch.cuda.get_device_name()}")

	gen_start_time = time.time()
	nwavs = model.generate(prompt_list)
	print(f"Generation took {time.time() - gen_start_time} seconds for {n_at_once}x {duration}s wavs using model {model_name} with params {model.generation_params}")

	filter_start_time = time.time()
	nwavs_cpu = [wav.detach().cpu() for wav in nwavs]
	nwavs_cpu_abs = [torch.abs(wav_cpu) for wav_cpu in nwavs_cpu]
	nwavs_max_amp = [torch.max(wav_abs) for wav_abs in nwavs_cpu_abs]
	nwavs_avg_amp = [torch.mean(wav_abs) for wav_abs in nwavs_cpu_abs]
	nwavs_995th_amp = [torch.quantile(wav_abs, 0.995) for wav_abs in nwavs_cpu_abs]
	# nwavs_filtered = [wav_cpu if max_amp >= MIN_ACCEPTABLE_AMPLITUDE and amp995 >= MIN_ACCEPTABLE_995TH_AMPLITUDE]
	nwavs_filtered_idx = []
	nwavs_silent_idx = []
	for i, (wav_cpu, max_amp, amp995) in enumerate(zip(nwavs_cpu, nwavs_max_amp, nwavs_995th_amp)):
		if max_amp >= MIN_ACCEPTABLE_AMPLITUDE and amp995 >= MIN_ACCEPTABLE_995TH_AMPLITUDE:
			nwavs_filtered_idx.append(i)
		else:
			nwavs_silent_idx.append(i)

	print(f"Filtered {len(nwavs_cpu) - len(nwavs_filtered_idx)} wavs below minimum amplitude threshold in {time.time() - filter_start_time} seconds")

	if resp_type == "single":
		if len(nwavs_filtered_idx) == 0:
			print("All wavs below minimum amplitude threshold, using best signal...")
			best_idx = torch.argmax(nwavs_995th_amp)
			wav = nwavs_cpu[best_idx]
		else:
			wav = nwavs_cpu[nwavs_filtered_idx[0]] # first filtered wav

		write_start_time = time.time()
		tmpfile = tempfile.NamedTemporaryFile(delete=False)
		audio_write(tmpfile.name, wav, model.sample_rate, format='wav', wav_fmt="pcm_u8", normalize=False, add_suffix=False)
		print(f"Writing took {time.time() - write_start_time} seconds")

		response = FileResponse(tmpfile.name, media_type="audio/wav")
		return response
	elif resp_type == "sorted":
		# sort by nwavs_avg_amp
		# nwavs_sorted = [wav_cpu for _, wav_cpu in sorted(zip(nwavs_avg_amp, nwavs_cpu), key=lambda pair: pair[0], reverse=True)]
		# sort by if nwavs_filtered
		nwavs_sorted_idxs = nwavs_filtered_idx + nwavs_silent_idx
		topnwavs_idx = nwavs_sorted_idxs[:sorted_top_n]
		# random.shuffle(topnwavs)

		write_wavs = [nwavs_cpu[i] for i in topnwavs_idx]
		write_prompt_list = [prompt_list[i] for i in topnwavs_idx]
	elif resp_type == "shuffled":
		random_perm = np.random.permutation(len(nwavs_cpu))
		write_wavs = [nwavs_cpu[i] for i in random_perm]
		write_prompt_list = [prompt_list[i] for i in random_perm]
	elif resp_type == "filtered":
		if len(nwavs_filtered_idx) == 0:
			print("All wavs below minimum amplitude threshold, returning all silent signals...")
			write_wavs = nwavs_cpu
			write_prompt_list = prompt_list
		else:
			write_wavs = [nwavs_cpu[i] for i in nwavs_filtered_idx]
			write_prompt_list = [prompt_list[i] for i in nwavs_filtered_idx]
	elif resp_type == "all":
		write_wavs = nwavs_cpu
		write_prompt_list = prompt_list
	else:
		raise ValueError("resp_type must be 'single', 'sorted', 'filtered', or 'all'")

	write_start_time = time.time()
	nwavs_b64 = []
	# for wav in write_wavs:
	# 	tmpfile = tempfile.NamedTemporaryFile(delete=False)
	# 	audio_write(tmpfile.name, wav, model.sample_rate, format='wav', wav_fmt="pcm_u8", normalize=False, add_suffix=False)
	# 	base64_str = base64.b64encode(tmpfile.read()).decode("ascii")
	# 	nwavs_b64.append(base64_str)
	for wav in write_wavs:
		wav_norm = normalize_audio(wav, normalize=(normalize_output and model_is_haptic), peak_normalize_db_clamp=-10, strategy="peak", sample_rate=model.sample_rate).numpy()
		if model_name in AUDIO_MODEL_IDS:
			wav_norm = amp_env_on_wav_norm(wav_norm, model.sample_rate, model_dict[HAPTIC_MODEL_IDS[0]].sample_rate)
		wav_u8_bytes = (wav_norm * 128 + 128).astype("uint8").tobytes()
		base64_str = base64.b64encode(wav_u8_bytes).decode("ascii")
		nwavs_b64.append(base64_str)
	print(f"Creating b64s took {time.time() - write_start_time} seconds")

	if create_variants:
		return { "wavs": nwavs_b64, "prompts": write_prompt_list }
	else:
		return { "wavs": nwavs_b64, "prompts": write_prompt_list }

WANTED_BIN_SIZE_SEC: float = 0.010 # 10ms
BASE_FREQ: float = 200.0 # 200Hz
def amp_env_on_wav_norm(wav_norm: np.ndarray, input_sample_rate: int, output_sample_rate: int):
	wav_norm = wav_norm.squeeze()
	num_samples = len(wav_norm)
	duration_sec = num_samples / input_sample_rate
	samples_per_bin = int(WANTED_BIN_SIZE_SEC * input_sample_rate)
	num_bins = num_samples // samples_per_bin
	# print(f"wav_norm.shape: {wav_norm.shape}, samples_per_bin: {samples_per_bin}, num_bins: {num_bins}")
	wav_chunks = np.array_split(wav_norm, num_bins)
	rms_bins = np.array([np.sqrt(np.mean(chunk ** 2)) for chunk in wav_chunks])

	assert num_bins == len(rms_bins), f"num_bins: {num_bins}, len(rms_bins): {len(rms_bins)}"
	rms_max = np.max(rms_bins)
	rms_norm = np.sqrt(2)
	rms_amplify = max(1.0, min(1.2, 1.0 / (rms_max * rms_norm)))
	rms_norm_amp = rms_norm * rms_amplify
	out_samples = int(duration_sec * output_sample_rate)

	phase_acc = 0.0
	output = np.zeros(out_samples)
	for i in range(out_samples):
		t = i / output_sample_rate
		t_prog = t / duration_sec

		bin_fi = t_prog * num_bins
		bin_lo = int(bin_fi)
		bin_hi = min(num_bins - 1, int(math.ceil(bin_fi)))
		bin_fr = bin_fi - bin_lo
		rms = (rms_bins[bin_lo] * (1.0 - bin_fr) + rms_bins[bin_hi] * bin_fr) * rms_norm_amp
		freq = 0.0 # no freq bins
		freq_offset = (rms - 0.3) * 100.0

		phase_delta = 2.0 * math.pi * (BASE_FREQ + freq_offset) / output_sample_rate # Numerically controlled oscillator (Direct digital synthesis), so can change freq without phase discontinuity
		phase_acc = (phase_acc + phase_delta) % (2.0 * math.pi)

		sample = rms * math.sin(phase_acc)

		output[i] = sample

	return output

import sys
if __name__ == "__main__":
	if len(sys.argv) == 2 and sys.argv[1] == "test":
		start_time = time.time()
		print("starting...")
		context = type("Context", (), {})()
		context.on_start_value = init_models()
		print(f"init took {time.time() - start_time} seconds")
		inf_start_time = time.time()
		inference(context, "dog barking")
		print(f"inference took {time.time() - inf_start_time} seconds")
		print(f"Done in {time.time() - start_time} seconds total")
	elif len(sys.argv) == 2 and sys.argv[1] == "testoai":
		oai_client = OpenAI()
		start_time = time.time()
		results = get_prompt_variants(oai_client, "dog barking")
		print(f"Took {time.time() - start_time} seconds for {results}")