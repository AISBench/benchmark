import os
import socket
import torch
import pickle

import torch.distributed


# ------------------------------------------------------- #
#                        distributed                      #
# ------------------------------------------------------- #
# Module-level device for all_gather etc. Set by dist_init(device=...).
_current_device = 'cuda'


def get_device():
    """Return current device string ('cuda' or 'npu') for tensor placement."""
    return _current_device


def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)


def dist_init(device=None):
    """Initialize distributed. device: 'cuda' | 'npu' | None (auto-detect)."""
    global _current_device
    if device is None:
        device = os.environ.get('VBENCH_DEVICE', '').lower() or None
    if device is None:
        if getattr(torch, 'npu', None) and torch.npu.is_available():
            device = 'npu'
        else:
            device = 'cuda'
    device = str(device).lower()
    if device not in ('cuda', 'npu'):
        device = 'cuda'
    _current_device = device

    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    # When MASTER_PORT is not preset, pick a free local TCP port instead of
    # hard-coding 29500. This avoids EADDRINUSE when multiple eval tasks run
    # concurrently with world_size=1 in each process.
    if 'MASTER_PORT' not in os.environ:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            free_port = s.getsockname()[1]
        os.environ['MASTER_PORT'] = str(free_port)
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'

    if os.name == 'nt':
        backend = 'gloo'
    else:
        backend = 'hccl' if device == 'npu' else 'nccl'
    torch.distributed.init_process_group(backend=backend, init_method='env://')
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if device == 'npu' and getattr(torch, 'npu', None):
        torch.npu.set_device(local_rank)
    else:
        torch.cuda.set_device(local_rank)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    dev = get_device()
    # serialized to a Tensor
    origin_size = None
    if not isinstance(data, torch.Tensor):
        buffer = pickle.dumps(data)
        storage = torch.ByteStorage.from_buffer(buffer)
        tensor = torch.ByteTensor(storage).to(dev)
    else:
        origin_size = data.size()
        tensor = data.reshape(-1)

    tensor_type = tensor.dtype

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to(dev)
    size_list = [torch.LongTensor([0]).to(dev) for _ in range(world_size)]
    torch.distributed.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.FloatTensor(size=(max_size,)).to(dev).to(tensor_type))
    if local_size != max_size:
        padding = torch.FloatTensor(size=(max_size - local_size,)).to(dev).to(tensor_type)
        tensor = torch.cat((tensor, padding), dim=0)
    torch.distributed.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        if origin_size is None:
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        else:
            buffer = tensor[:size]
            data_list.append(buffer)

    if origin_size is not None:
        new_shape = [-1] + list(origin_size[1:])
        resized_list = []
        for data in data_list:
            # suppose the difference of tensor size exist in first dimension
            data = data.reshape(new_shape)
            resized_list.append(data)

        return resized_list
    else:
        return data_list


def barrier():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

# ------------------------------------------------------- #


def merge_list_of_list(results):
    results = [item for sublist in results for item in sublist]
    return results


def gather_list_of_dict(results):
    results = all_gather(results)
    results = merge_list_of_list(results)
    return results


def distribute_list_to_rank(data_list):
    data_list = data_list[get_rank()::get_world_size()]
    return data_list
