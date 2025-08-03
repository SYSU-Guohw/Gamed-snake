import sys
sys.argv.extend(['--cfg_file', 'configs/sbd_snake.yaml', 'ct_score', '0.4', 'train_or_test', 'test'])

from run import run_test_medical
run_test_medical()