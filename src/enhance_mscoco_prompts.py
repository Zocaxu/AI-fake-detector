import os
import numpy as np

MEDIUM = ['A photography of', 'A digital painting of', 'A 3D rendering of', 'An oil painting of']
STYLE = ['hyperrealistic', 'surrealist', 'realistic', 'lifelike', 'natural']
ENHANCING = ['highly detailed', 'sharp focus', 'stunning details', 'true to real life', 'stunningly beautiful', 'masterpiece collection', 'bright colors', 'realistic colors']
LIGHTING = ['bright lighting', 'dark lighting', 'cimenatic lighting', 'sun lighting', 'volumetric lighting']

prompt_file_path = '/vol/bitbucket/mb322/AI-fake-detection/mscoco_prompts/gather_mscoco_5000'
output_file_path = '/vol/bitbucket/mb322/AI-fake-detection/mscoco_prompts_enhanced/enhanced_mscoco_5000'
n_files = 10

for i in range(1,n_files+1):
    with open(prompt_file_path + f"_{i}.txt", 'r') as f:
        prompts = f.readlines()
        r = open(output_file_path + f"_{i}.txt", 'w')
        for prompt in prompts:
            if prompt[-1] == '\n':
                prompt = prompt[0:-1]
            if prompt[-1] == ' ':
                prompt = prompt[0:-1]
            if prompt[-1] == '.':
                prompt = prompt[0:-1]
            medium = np.random.choice(MEDIUM)
            style = np.random.choice(STYLE)
            enhancings = np.random.choice(ENHANCING, 2)
            lighting = np.random.choice(LIGHTING)
            enhanced_prompt = f"{medium} {prompt}, {enhancings[0]}, {enhancings[1]}, {lighting}"
            r.write(enhanced_prompt + '\n')
        r.close()

        