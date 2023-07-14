from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

config = Cfg.load_config_from_name('vgg_transformer')

config['vocab'] = 'tập vocab'

dataset_params = {
    'name':'hw',
    'data_root':'./data_line/',
    'train_annotation':'train_line_annotation.txt',
    'valid_annotation':'test_line_annotation.txt'
}

params = {
         'print_every':2,
         'valid_every':15*200,
          'iters':60000,
          'checkpoint':'./checkpoint/transformerocr_checkpoint.pth',
          'export':'./weights/transformerocr.pth',
          'metrics': 10000,
          'batch_size': 32
         }

config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['device'] = 'cpu'

trainer = Trainer(config, pretrained=True)

trainer.visualize_dataset()

trainer.train()

trainer.visualize_prediction()

trainer.config.save('config.yml')