import os
import argparse
from torch.backends import cudnn
from dataset import get_loader
from solver import Solver_whaformer





def main(args):
    cudnn.benchmark = True

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    if args.result_fig:
        fig_path = os.path.join(args.save_path, 'fig')
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
            print('Create path : {}'.format(fig_path))

    data_loader = get_loader(mode=args.mode,
                             load_mode=args.load_mode,
                             saved_path=args.saved_path,
                             test_patient=args.test_patient,
                             patch_n=(args.patch_n if args.mode=='train' else None),
                             patch_size=(args.patch_size if args.mode=='train' else None),
                             transform=args.transform,
                             batch_size=(args.batch_size if args.mode=='train' else 1),
                             num_workers=args.num_workers)

    data_loader_test = get_loader(mode='test',
                             load_mode=args.load_mode,
                             saved_path=args.saved_path,
                             test_patient=args.test_patient,
                             patch_n=None,
                             patch_size=None,
                             transform=args.transform,
                             batch_size=1,
                             num_workers=args.num_workers,
                             shuffle=False)

    solver = Solver_whaformer(args, data_loader, data_loader_test)

    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()



    parser.add_argument('--mode', type=str, default='test', help="train | test")
    parser.add_argument('--saved_path', type=str, default='../npy_img_1mm_all/')
    parser.add_argument('--test_patient', type=str, default='L506')

    parser.add_argument('--test_epoch', type=int, default=106)



    parser.add_argument('--val_epoch', type=int, default=103)

    parser.add_argument('--save_path', type=str, default="save/whaformer/", help="save path")
    parser.add_argument('--load_mode', type=int, default=0, help="0 | 1")
    parser.add_argument('--patch_n', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=64)
    parser.add_argument('--transform', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--dim', type=int, default=32)

    # set_model
    parser.add_argument("--init_lr", default=8e-5, type=float)

    # train
    parser.add_argument('--num_epochs', type=int, default=121)
    parser.add_argument('--print_iters', type=int, default=100)
    parser.add_argument('--decay_iters', type=int, default=54400)

    parser.add_argument('--num_workers', type=int, default=7)
    parser.add_argument('--device', type=str)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--n_d_train', type=int, default=1)

    # test
    parser.add_argument('--save_fig_iters', type=int, default=1)

    parser.add_argument('--result_fig', type=bool, default=True)
    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=3072.0)
    parser.add_argument('--trunc_min', type=float, default=-160.0)
    parser.add_argument('--trunc_max', type=float, default=240.0)

    parser.add_argument('--lambda_', type=float, default=10.0)



    args = parser.parse_args()
    main(args)