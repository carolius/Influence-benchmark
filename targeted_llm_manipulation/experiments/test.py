import targeted_llm_manipulation.experiments as exp
from targeted_llm_manipulation.data_root import PROJECT_DATA
from targeted_llm_manipulation.RL.training_funcs import print_accelerator_info
from targeted_llm_manipulation.root import PROJECT_ROOT
from targeted_llm_manipulation.utils.utils import *


def main():
    # Save a file with hello world in it
    print("Test")
    with open(PROJECT_DATA / "hello.txt", "w") as f:
        f.write(f"{PROJECT_ROOT}")


if __name__ == "__main__":
    main()
