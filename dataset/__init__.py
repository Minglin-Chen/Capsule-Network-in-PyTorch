import torchvision
import torchvision.transforms as T

def dataset_provider(name, dataset_root, is_train=True):

    dataset = None

    if name == 'MNIST':

        transforms_list = []
        if is_train:
            transforms_list.append( T.RandomAffine(degrees=0.0, translate=(2.5/28, 2.5/28)) )
        transforms_list.append( T.ToTensor() )
        transforms_list.append( T.Normalize((0.5,), (0.5,)) )

        dataset = torchvision.datasets.MNIST(
                root=dataset_root,
                train=is_train,
                transform=T.Compose(transforms_list), 
                download=True)

    return dataset