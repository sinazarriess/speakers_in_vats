import torch

# https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

checkpoint = '/Users/sina/research/a-PyTorch-Tutorial-to-Image-Captioning/pretrained/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'  # model checkpoint
checkpoint = torch.load(checkpoint,map_location=torch.device('cpu'))
decoder = checkpoint['decoder']
decoder = decoder.to(device)
