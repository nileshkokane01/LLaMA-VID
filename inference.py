import argparse
import torch

from llamavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llamavid.conversation import conv_templates, SeparatorStyle
from llamavid.model.builder import load_pretrained_model,load_pretrained_model_dummy
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
import numpy as np




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

import torch.nn as nn


def init_weights(self):
    for name, param in self.named_parameters():
        if 'weight' in name:
            if param.dim() > 1:  # Check if the parameter is a weight tensor
                nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)

# Example model with language model and projection layers



def main(args):
    # Model
    disable_torch_init()

    #model_name = get_model_name_from_path('work_dirs\llama-vid-7b-full-336')
    model_name = get_model_name_from_path('work_dirs/llama-vid/llama-vid-video')
   
    print(model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model('work_dirs/llama-vid/llama-vid-video', args.model_base, model_name, args.load_8bit, args.load_4bit)
    #print('tokeenizer : ' , tokenizer ) 

    #model.load_state_dict(torch.load('work_dirs\\llama-vid-7b-full-336\\model.pth'))


	


    #init_weights(model)


  
    #model.save('nilesh_copy.pth')

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
        
    elif "v1" in model_name.lower() or "vid" in model_name.lower():
        conv_mode = "llava_v1"
        
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
        
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles
    
    args.image_file="test.mp4.mp4"

    if args.image_file is not None:
        if '.mp4' in args.image_file:
            image = load_video(args.image_file)
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
            image_tensor = [image_tensor]
        else:
            image = load_image(args.image_file)
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
    else:
        image_tensor = None




  
    inp  ="<image>\nUSER: What's the content of the image?\nASSISTANT:"

    print(f"{roles[1]}: ", end="")

    model.update_prompt([[inp]])

    if args.image_file is not None:
        # first message
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        image = None
    else:
        # later messages
        conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    prompt="<image>\nUSER: What's the content of the image?\nASSISTANT:"

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    print('====================================================')
    print('PROMPT : ' , prompt )     
    print('MODEL PROMPT : ' , model.get_prompt())
    print('====================================================')
    

    
   
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria])
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip() 
    

    
    


    outputs = "%the output of the model "
    
    conv.messages[-1][-1] = outputs
    conv.messages[-2][-1] = conv.messages[-2][-1].replace(DEFAULT_IMAGE_TOKEN+'\n','')

    if args.debug:
        print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


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
