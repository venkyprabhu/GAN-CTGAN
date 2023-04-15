import produce_gan as f1
import yaml

config_file='config.yml'

try:
    with open(config_file) as yml_data:
        conf = yaml.safe_load(yml_data)
except FileNotFoundError:
    conf = dict()

file_dir= conf['file_dir']
gan_rows= conf['gan_rows']


gan_output,gan_output1=f1.run_ctgan(file_dir, gan_rows)


print("GAN generated at ", gan_output)
