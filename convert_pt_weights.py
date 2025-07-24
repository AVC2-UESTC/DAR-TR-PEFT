
import torch
import os

def main():
    # load the weights
    weights_path = './model_weights/dinov2_vitb14_pretrain.pth'
    assert os.path.exists(weights_path), f'{weights_path} does not exist!'
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
    
    for k,v in checkpoint.items():
        print(k)
        
    print('------------------------')
        
    new_checkpoint = dict()
    for k,v in checkpoint.items():
        if k == 'mask_token':
            continue
        else:
            new_name = 'backbone.' + k
            new_checkpoint[new_name] = v
    
    # for k,v in new_checkpoint.items():
    #     print(k, v.shape)
    
    torch.save(new_checkpoint, 'model_weights/dinov2_vitb14_pretrain_new.pth')
    print('done!')
    
    
if __name__ == '__main__':
    main()






