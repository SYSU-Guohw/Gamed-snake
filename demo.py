import sys
sys.argv.extend(['--cfg_file', 'configs\sbd_snake.yaml', 'ct_score', '0.5'])

from run import run_demo
run_demo()