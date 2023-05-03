import fire
from dataset2metadata.process import process


def main():
    fire.Fire(process)

if __name__ == "__main__":
    main()