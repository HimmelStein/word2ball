import word2ball
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--restart", type=int, default=1,
                        help='whether need to restart: 0 for not restart, 1 for restart')
    args = parser.parse_args()
    print(args)
    restartTF = args.restart
    print(restartTF , "restartTF ")
    word2ball.training_balls(restart=restartTF)
