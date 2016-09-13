
> Repo for Speech Synth. work <br>
> Author: Rithesh Kumar

wavenet.py --> Implementation of DeepMind's wavenet paper. This file contains the architecture and the code to start training <br>
<br>
lib --> Library where I've added custom layers coded in Theano. This is built on top of Ishaan's library structure. <br>
<br>
dataset.py --> Loads the files from /data/lisatmp3/kumarrit/blizzard where there are ~140721 speech sections from Blizzard dataset chunked into smaller (8 second) sections. I have implemented mewlaw quantization, but temporarily do not use that. 
