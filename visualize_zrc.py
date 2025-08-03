import sys
sys.argv.extend(['--cfg_file', 'configs\sbd_snake.yaml', 'ct_score', '0.5', 'vis_zrc', '1'])

from run import run_visualize
run_visualize()

# 中间结果可视化（显示检测方框）： E:\PyCharm-ZRC\PyCharm-Project\DeepSnake_remove_c\lib\datasets\voc\snake.py  __getitem__() 函数： visualize_utils.visualize_snake_detection(orig_img, ret)

# 进化后结果可视化（显示检测方框）： E:\PyCharm-ZRC\PyCharm-Project\DeepSnake_remove_c\lib\datasets\voc\snake.py  __getitem__() 函数： visualize_utils.visualize_snake_evolution(orig_img, ret)