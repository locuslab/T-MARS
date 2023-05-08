import fire
from dataset2metadata.process import process
import torch

torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    fire.Fire(process)

if __name__ == "__main__":
    main()
