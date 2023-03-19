import os.path as osp
import basicsr
import odisr

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    basicsr.train_pipeline(root_path)
