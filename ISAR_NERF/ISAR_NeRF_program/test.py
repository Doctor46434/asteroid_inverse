import json
import os
from opt import config_parser

config_parser = config_parser()
args = config_parser.parse_args()
config_params = vars(args)
if os.path.exists("model_parameter") is False:
    os.makedirs("model_parameter")

config_file = args.config_path
with open(config_file, "w") as f:
    json.dump(config_params, f, indent=4)