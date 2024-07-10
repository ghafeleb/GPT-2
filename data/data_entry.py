
data_choice = ['super_tiny_shakespear', 'tiny_shakespear']


def get_dataset_by_type(args, is_train=True, transform=None):
    if args.data_type in ['super_tiny_shakespear', 'tiny_shakespear']:
        with open('../data/input.txt', 'r') as file:
            dataset = file.read()
    return dataset