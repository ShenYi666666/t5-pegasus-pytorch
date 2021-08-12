from transformers import MT5Config, MT5ForConditionalGeneration, load_tf_weights_in_t5
import torch

config_path = 'E:\\预训练模型\\chinese_t5_pegasus_base\\config.json'
#config = MT5Config.from_pretrained('config.json')
config = MT5Config.from_pretrained(config_path)
model = MT5ForConditionalGeneration(config)


ckpt = 'E:\\预训练模型\\chinese_t5_pegasus_base\\model.ckpt'
#checkpoint_path = 'E:\\预训练模型\\chinese_t5_pegasus_base\\model.ckpt'
model = load_tf_weights_in_t5(model, config, ckpt)

torch.save(model.state_dict(), 'pytorch_model.bin')