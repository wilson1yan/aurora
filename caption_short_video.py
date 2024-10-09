import os
import os.path as osp
import argparse
import io
import time
import json
import pickle
import tarfile
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForCausalLM, CLIPImageProcessor
from huggingface_hub import snapshot_download

import bag
from src.xtuner.xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, PROMPT_TEMPLATE
from src.xtuner.xtuner.model.aurora import AuroraEncoder, AuroraModel
from src.xtuner.xtuner.tools.load_video import read_video_bytes_pyav


def bag_file_iterator(bag_files):
  for bag_file in bag_files:
    f = bag_file.open('rb')
    len_ = bag.read_len(f)
    idx = 0
    while idx < len_:
      md = pickle.loads(bag.read_data(f, idx))
      idx += 1
      n_clips = len(md['clip_md'])
      for j in range(n_clips):
        video_bytes = bag.read_data(f, idx + j)
        info = {'url': md['url'], 'clip': md['clip_md'][j]}
        yield (info, video_bytes)
      idx += n_clips


def process_text(inputs, tokenizer):
    chunk_encode = []
    for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
        if idx == 0:
            cur_encode = tokenizer.encode(chunk)
        else:
            cur_encode = tokenizer.encode(chunk, add_special_tokens=False)
        chunk_encode.append(cur_encode)
    # assert len(chunk_encode) == 2 # for single image
    ids = []
    for idx, cur_chunk_encode in enumerate(chunk_encode):
        ids.extend(cur_chunk_encode)
        if idx != len(chunk_encode) - 1:
            ids.append(IMAGE_TOKEN_INDEX)
    ids = torch.tensor(ids).cuda().unsqueeze(0)
    return ids


def process_input(video_bytes, image_processor, tokenizer, device):
  data = dict()
  video_frames = read_video_bytes_pyav(io.BytesIO(video_bytes), args.num_frm)
  image_tensor = image_processor(video_frames, return_tensors='pt')['pixel_values']
  image_tensor = [_image.to(dtype=torch.float16).cuda() for _image in image_tensor]
  data["pixel_values"] = torch.stack(image_tensor).unsqueeze(0)
  image_tokens = [DEFAULT_IMAGE_TOKEN] * len(video_frames)
  image_tokens = " ".join(image_tokens)
  text_input = image_tokens + "\n" + args.prompt
  prompt_text = PROMPT_TEMPLATE.vicuna["INSTRUCTION"].format(input=text_input, round=1)
  data["input_ids"] = process_text(prompt_text, tokenizer).to(device)
  return data


def main():
  #rank, size, local_rank = utils.setup_distributed()
  rank = int(os.getenv('SLURM_PROCID'))
  size = int(os.getenv('SLURM_NTASKS'))
  local_rank = 0
  print(f'Started {rank} / {size}')
  torch.cuda.set_device(local_rank)
  device = torch.device(f'cuda:{local_rank}')

  pretrained_pth = snapshot_download(repo_id=args.model_path) if not osp.isdir(args.model_path) else args.model_path
  pretrained_vit = osp.join(pretrained_pth, "visual_encoder")
  projector_path = osp.join(pretrained_pth, "projector")

  auroracap = AuroraModel(
      llm=AutoModelForCausalLM.from_pretrained(
          pretrained_model_name_or_path=pretrained_pth,
          trust_remote_code=True,
          torch_dtype=torch.float16,
      ),
      visual_encoder=AuroraEncoder.from_pretrained(
          pretrained_model_name_or_path=pretrained_vit,
          torch_dtype=torch.float16,
      ),
  ).to(device)
  auroracap.projector = AutoModel.from_pretrained(projector_path, torch_dtype=torch.float16, trust_remote_code=True).to(device)
  image_processor = CLIPImageProcessor.from_pretrained(
      pretrained_model_name_or_path="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",  # use standard CLIP processor
      trust_remote_code=True,
      size=378,
      crop_size=378,
  )
  tokenizer = AutoTokenizer.from_pretrained(
      pretrained_model_name_or_path=pretrained_pth,
      trust_remote_code=True,
      padding_side='right',
  )

  data_folder = Path(args.data_folder)
  output_folder = Path(args.output_folder)
  chunk_infos = json.load((output_folder / 'chunk_md' / f'{rank:09}.json').open('r'))
  output_folder = output_folder / f'{rank}'
  output_folder.mkdir(exist_ok=True, parents=True)

  n_done, n_error = 0, 0
  start = time.time()
  for j, (chunk_id, chunk_files) in enumerate(chunk_infos):
    chunk_files = [data_folder / name for name in chunk_files]
    src_tar_path = output_folder / 'tmp.tar'
    src_tar_path.unlink(missing_ok=True)
    tar_file = tarfile.open(str(src_tar_path), 'w')
    for i, (info, video_bytes) in enumerate(bag_file_iterator(chunk_files)):
      id = f'{i:09}'

      try:
        inp = process_input(video_bytes, image_processor, tokenizer, device)
      except Exception as e:
        print(f'Error processing video: {e}')
        n_done += 1
        n_error += 1
        continue

      auroracap.visual_encoder.reset_tome_r(args.token_kept_ratio)
      output = auroracap(inp, mode="inference")
      cont = auroracap.llm.generate(
          **output,
          do_sample=False,
          temperature=args.temperature,
          top_p=args.top_p,
          num_beams=args.num_beams,
          max_new_tokens=args.max_new_tokens,
      )
      caption = tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
      info['caption'] = caption

      info_bytes = json.dumps(info).encode('utf-8')
      tarinfo = tarfile.TarInfo(f"{id}.json")
      tarinfo.size = len(info_bytes)
      tar_file.addfile(tarinfo, fileobj=io.BytesIO(info_bytes))

      tarinfo = tarfile.TarInfo(f"{id}.mp4")
      tarinfo.size = len(video_bytes)
      tar_file.addfile(tarinfo, fileobj=io.BytesIO(video_bytes))
      n_done += 1
      spv = (time.time() - start) / n_done
      print(f'Completed {j + 1} / {len(chunk_infos)}, {spv:.3f} SPV')
    tar_file.close()
    dst_tar_path = output_folder / f'{chunk_id}.tar'
    os.system(f'mv {src_tar_path} {dst_tar_path}')
  print('done', rank)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_folder', type=str, required=True)
  parser.add_argument('--output_folder', type=str, required=True)

  parser.add_argument('--model_path', type=str, help='path to the model', default='wchai/AuroraCap-7B-VID-xtuner')
  parser.add_argument('--prompt', type=str, help='prompt for the model', default='Describe the video in detail.')
  parser.add_argument('--visual_input', type=str, help='path to the video or image file', default='output.png')
  parser.add_argument('--num_frm', type=int, help='number of frames to sample from the video', default=8)
  parser.add_argument('--token_kept_ratio', type=float, help='token merge ratio', default=0.8)
  parser.add_argument('--temperature', type=float, help='temperature', default=0.0)
  parser.add_argument('--top_p', type=float, help='top p', default=1.0)
  parser.add_argument('--num_beams', type=int, help='number of beams', default=1)
  parser.add_argument('--max_new_tokens', type=int, help='max new tokens', default=512)

  args = parser.parse_args()
  main()
