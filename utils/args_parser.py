import argparse


def tempzero_argparser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--per_frame', type=int, default=2,
                        help='+- per frame')
    parser.add_argument('--gaussian_attack', type=bool, default=False,
                        help='gaussian attack')
    parser.add_argument('--fps', type=int, default=60,
                        help='save video fps')
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='epsilon for fgsm')
    parser.add_argument('--train_folder', type=str, default='C:/Users/user/Downloads/kinetics400_validation/val_256_120fps/',
                    help='input video')
    parser.add_argument('--targeted_attack', type=bool, default=False,
                    help='input video')
    parser.add_argument('--logger_name', type=str, default='logFile')
    parser.add_argument('--our_method', type=str, default='temp_zero')
    parser.add_argument('--c_value', type=float, default=1.4)
    parser.add_argument('--save_foler', type=str, default='C:/Users/user/Desktop/TempZero/runs/')

    # pure nes, sub_num....
    args = parser.parse_args()

    return args


# if __name__ == '__main__':
#     args = tempzero_argparser()
#     print(args)