import argparse
import torch

from llamavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llamavid.conversation import conv_templates, SeparatorStyle
from llamavid.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
from decord import VideoReader, cpu

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import inspect
import inspect

def print_file_and_line_info():
    frame_info = inspect.stack()[1]
    file_name = frame_info.filename
    line_number = frame_info.lineno
    print("File:", file_name, "Line:", line_number)


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def load_video(video_path, fps=1):
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames

def main(args):
    # Model
    disable_torch_init()

    print(args)
    print("---------------model_path==============")
    print(args.model_path)
    
    print('model_name' , model_name)

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2) # set to 0.5 for video
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)