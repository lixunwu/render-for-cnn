import os

pascal3d_root = '/home/wlx/dataset/PASCAL3D+_release1.1'

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
render4cnn_weights = os.path.join(root_dir, 'model_weights/r4cnn.pkl')

