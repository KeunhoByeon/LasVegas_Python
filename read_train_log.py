import os


def read_log(txt_path):
    now_epoch, best_epoch, best_acc, best_confusion_mat = 0, 0, 0, None
    with open(txt_path, 'r') as rf:
        args = rf.readline()
        for line in rf.readlines():
            line_split = line.split('  ')
            if '*Validation' in line:
                acc = float(line_split[1].split(': ')[1])
                if acc > best_acc:
                    best_epoch = now_epoch
                    best_acc = acc
            else:
                now_epoch = int(line_split[0].split(' ')[0].split('/')[0][1:])

        print('[{}] best epoch: {}/{}  best acc: {}  best confusion mat: {}'.format(txt_path, best_epoch, now_epoch, best_acc, best_confusion_mat))
        print('Arguments: {}'.format(args.replace('Namespace(', '')[:-2]))
        print()
    return best_epoch, now_epoch, best_acc


if __name__ == '__main__':
    base_dir = './results/'
    for test_name in sorted(os.listdir(base_dir)):
        txt_path = os.path.join(base_dir, test_name, 'log.txt')
        if not os.path.isfile(txt_path):
            continue
        read_log(txt_path)
