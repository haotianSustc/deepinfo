# DeepInfo 

This is the code of paper deepinfo.

### How to run this:

run the following command:

```bash
python main.py --datasets_name=CUB --net_name=vgg16 --step_name=net;
python main.py --datasets_name=CUB --net_name=vgg16 --step_name=x;
```

### main.py:

important parameters:
* datasets_name:to select the dataset，used in tools.get_input.
* out_planes/image_size：parameters depending on datasets_name。
* net_name:to select the evaluated neural network，used in tools.get_input。
* net_lr/net_batch_size/...：parameters to train the evaluated neural network.
* step_name: "net" for network training, x for SID compuation, decoder for decoder training, y for RU computation.
* capability_batch_size: to control the batch size of noise in SID/RU computation. If it's very small, the result will not be stable. It can't be very large, limited to GPU memory.
* capability_lambda_init_x/capability_lambda_init_y: It's the lambda parameter shown in our paper, which can balance two kinds of loss function. It can't be either very large or very small.

TODO: add datasets and trained net/decoder

TODO: give more explanations to all the parameters.

