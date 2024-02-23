from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom

map_dict = {
    'ETTh': Dataset_ETT_hour,
    'ETTm': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    if flag == 'test':
        shuffle_flag = False
        batch_size = 1
    else:
        shuffle_flag = True
        batch_size = args.batch_size

    data_reader = map_dict[args.data_reader]
    data_set = data_reader(
        args=args,
        flag=flag
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=True
    )
    args.logger.info('Mode: {}, Sample Num: {}, Batch Num: {}'.format(flag, len(data_set), len(data_loader)))
    return data_set, data_loader
