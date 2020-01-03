install `nest`
```bash
pip install git+https://github.com/ZhouYanzhao/Nest.git
```

use `nest` to install `s3n` modules
```bash
# out the S3N git project and install, namespace is s3n 
nest module install ./S3N/ s3n  # namespace name

# check install results
# each register function is a module here, which can set in *.yml with module name
$ nest module list --filter s3n
24 Nest modules found.
[0] s3n.adadelta_optimizer (version)
[1] s3n.best_meter (version)
[2] s3n.checkpoint (version)
[3] s3n.cross_entropy_loss (version)
[4] s3n.fetch_data (version)
[5] s3n.fgvc_dataset (version)
[6] s3n.finetune (version)
[7] s3n.ft_resnet (version)
[8] s3n.image_transform (version)
[9] s3n.interval (version)
[10] s3n.loss_meter (version)
[11] s3n.multi_smooth_loss (version)
[12] s3n.multi_topk_meter (version)
[13] s3n.network_trainer (version)
[14] s3n.print_state (version)
[15] s3n.rle_decode (version)
[16] s3n.rle_encode (version)
[17] s3n.s3n (version)
[18] s3n.sgd_optimizer (version)
[19] s3n.smooth_loss (version)
[20] s3n.three_stage (version)
[21] s3n.topk_meter (version)
[22] s3n.update_lr (version)
[23] s3n.vis_trend (version)
```

uninstall the module

```bash
# given namespace
$ nest module remove s3n
# given path to the namespace
$ nest module remove ./S3N
```

module namespace

```bash
pprint(modules.namespaces)
# {'s3n': {'module_path': '/nfs/xs/Codes/fine-grain/S3N'}}
```

do the experiments
```bash
# run baseline
$ PYTHONWARNINGS='ignore' CUDA_VISIBLE_DEVICES=0,1 nest task run ./demo/cub_baseline.yml
# run S3N
$ PYTHONWARNINGS='ignore' CUDA_VISIBLE_DEVICES=0,1 nest task run ./demo/cub_s3n.yml
```

the structure of experiment `*.yml` is very simple, each indent param is just the module(function) param.