## schedulers.py

### Scheduler_Config

**Format of the config file**

```json
{
    "Scheduler_cfg": {
        "scheduler_cfg_name": "Scheduler_Config",
        "scheduler_cfg_args": {
            "device": "{The device: cuda or cpu}",
            "seed": "{The random seed}",
            "pretrained": "{The path to the pretrained weights}"
            "batch_size": "{The batch size}",
            "weight_decay": "{The weight decay}",
            "epochs": "{The number of epochs}",
            "warmup_epochs": "{The number of warmup epochs}",
            "eval_interval": "{The interval for evaluation}",
            "lr": "{The initial learning rate}",
            "print_freq": "{The frequency for printing information}",
            "resume": "{The path to the resume checkpoint}",
            "fine_tune_w": "{The path to the fine tune checkpoint}",
            "start_epoch": "{The starting epoch for resume}",
            "amp": "{Whether to use automatic mixed precision}",
            "metrics_cfg": "{The metrics configuration}"
        }
    }
}
```

Here is an example of Scheduler_Config
```json
{
    "Scheduler_cfg": {
        "scheduler_cfg_name": "Scheduler_Config",
        "scheduler_cfg_args": {
            "device": "cuda",
            "seed": 3407,
            "pretrained": "./mit_b4.pth",
            "batch_size": 20,
            "weight_decay": 1e-4,
            "epochs": 20,
            "warmup_epochs": 3,
            "eval_interval": 2,
            "lr": 0.0015,
            "print_freq": 50,
            "resume": null,
            "fine_tune_w": null,
            "start_epoch": 0,
            "amp": false,
            "metrics_cfg": [
                {
                    "metric_type": "MAE", 
                    "resize_logits": true, 
                    "name": "MAE"
                }, 
                {
                    "metric_type": "Fmeasure", 
                    "resize_logits": true, 
                    "name": "max_F", 
                    "metric_args": {
                        "beta_sq": 0.3,
                        "mode": "max"
                    }
                }
            ]
        }
    }
}

```






