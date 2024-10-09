import random
from dataclasses import dataclass
from typing import List, Optional, Type

# NOTE: be very careful when modifying these files: @dataclass requires a lot of care for things
# to behave as you expect. Often you need to add a lot of type hints to make things work as expected.


@dataclass
class AccelerateConfig:

    mixed_precision: str = "bf16"
    num_machines: int = 1
    rdzv_backend: str = "static"
    same_network: bool = True
    main_training_function: str = "main"
    machine_rank: int = 0
    num_processes: Optional[int] = None
    gpu_ids: Optional[List[int]] = None
    dynamo_backend: str = "no"
    gradient_accumulation_steps: int = 16

    def set_gpu_ids(self, gpu_ids: Optional[List[int]]):
        if gpu_ids is None:
            return
        self.gpu_ids = gpu_ids[:1]  # A normal accelerate config only supports one GPU without FSDP
        self.num_processes = len(self.gpu_ids)
        print(f"Accelerate training on GPUs: {self.gpu_ids}")

    def update_gradient_accumulation_steps(self, effective_batch_size: int, per_device_train_batch_size: int):
        """
        Update gradient accumulation steps based on the given batch size and number of processes.

        Args:
            batch_size (int): The desired batch size.

        Returns:
            None
        """
        if effective_batch_size % per_device_train_batch_size != 0:
            effective_batch_size -= effective_batch_size % per_device_train_batch_size
            print(
                f"Warning: effective_batch_size is not evenly divisible by per_device_train_batch_size. Using effective_batch_size {effective_batch_size}"
            )
        batch_size = effective_batch_size // per_device_train_batch_size

        if not self.num_processes:
            print(f"Warning: num_processes is not set. Using gradient_accumulation_steps of {batch_size}.")
            self.gradient_accumulation_steps = batch_size
            return

        if batch_size % self.num_processes != 0:
            adjusted_batch_size = batch_size - (batch_size % self.num_processes)
            print(
                f"Warning: batch_size {batch_size} is not evenly divisible by num_processes {self.num_processes}. "
                f"Adjusting to {adjusted_batch_size}."
            )
            batch_size = adjusted_batch_size

        self.gradient_accumulation_steps = max(batch_size // self.num_processes, 1)
        print(f"Set gradient_accumulation_steps to {self.gradient_accumulation_steps}")

    def to_cli_args(self):
        assert self.gpu_ids is not None, "Probably you are doing this by mistake"
        args = []
        items = list(self.__dict__.items())
        for k, v in items:
            if k == "gradient_accumulation_steps":
                continue
            if isinstance(v, bool):
                if "16bit" in k:
                    args.append(f"--{k.replace('_', '-')}={v}")
                elif v:
                    args.append(f"--{k.replace('_', '-')}")
            elif isinstance(v, List):
                args.append(f"--{k.replace('_', '-')}={','.join(map(str, v))}")
            else:
                args.append(f"--{k.replace('_', '-')}={v}")
        return args


@dataclass
class AccelerateConfigFSDP(AccelerateConfig):

    enable_cpu_affinity: bool = False
    use_cpu: bool = False

    fsdp_auto_wrap_policy: str = "TRANSFORMER_BASED_WRAP"
    fsdp_backward_prefetch: str = "BACKWARD_PRE"
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_state_dict_type: str = "FULL_STATE_DICT"

    fsdp_activation_checkpointing: bool = False
    fsdp_sync_module_states: bool = True
    fsdp_use_orig_params: bool = True
    fsdp_cpu_ram_efficient_loading: bool = True
    fsdp_forward_prefetch: bool = False
    fsdp_offload_params: bool = False

    def set_gpu_ids(self, gpu_ids: Optional[List[int]]):
        if gpu_ids is None:
            return
        self.gpu_ids = gpu_ids
        self.num_processes = len(self.gpu_ids)
        print(f"Accelerate training on GPUs: {self.gpu_ids}")

    def to_cli_args(self):
        assert self.gpu_ids is not None, "Probably you are doing this by mistake"
        args = ["--use_fsdp"]
        items = list(self.__dict__.items())
        print(items)
        for k, v in items:
            if isinstance(v, bool):
                if "fsdp" or "16bit" in k:
                    args.append(f"--{k.replace('_', '-')}={v}")
                elif v:
                    args.append(f"--{k.replace('_', '-')}")

            elif isinstance(v, List):
                args.append(f"--{k.replace('_', '-')}={','.join(map(str, v))}")
            else:
                args.append(f"--{k.replace('_', '-')}={v}")
        return args


@dataclass
class AccelerateConfigDeepSpeed(AccelerateConfig):
    enable_cpu_affinity: bool = False
    use_cpu: bool = False
    use_deepspeed: bool = True

    gradient_clipping: float = 1.0
    offload_param_device: Optional[str] = None
    offload_optimizer_device: Optional[str] = None

    # We want each version of the script that is currently running on a machine to be using a different port, or that leads to crashes.
    # This is hacky, but the principled way to do this seems broken https://github.com/bmaltais/kohya_ss/issues/2138
    main_process_port: int = (lambda n: n - (n // 10000 in [0, 5]))(random.randint(10000, 65535))

    def set_gpu_ids(self, gpu_ids: Optional[List[int]]):
        if gpu_ids is None:
            return
        self.gpu_ids = gpu_ids
        self.num_processes = len(self.gpu_ids)
        print(f"Accelerate training on GPUs: {self.gpu_ids}")

    def set_gradient_clipping(self, max_grad_norm: float):
        self.gradient_clipping = max_grad_norm


@dataclass
class AccelerateConfigDeepSpeed1(AccelerateConfigDeepSpeed):
    zero_stage: int = 1


@dataclass
class AccelerateConfigDeepSpeed2(AccelerateConfigDeepSpeed):
    zero_stage: int = 2


@dataclass
class AccelerateConfigDeepSpeed3(AccelerateConfigDeepSpeed):
    zero_stage: int = 3
    zero3_save_16bit_model: bool = True


def get_accelerate_config_mapping() -> dict[str, Type[AccelerateConfig]]:
    mapping = {}

    # Add all subclasses of AccelerateConfig to the mapping, this also includes the subclasses of the subclasses.
    # This allows us to specify any subclass of AccelerateConfig by its name in the experiment config.
    def add_subclasses(cls):
        for subclass in cls.__subclasses__():
            key = subclass.__name__.replace("AccelerateConfig", "")
            mapping[key] = subclass
            add_subclasses(subclass)  # Recursively add subclasses

    add_subclasses(AccelerateConfig)

    # Add the base AccelerateConfig class with the key "Single_GPU"
    mapping["Single_GPU"] = AccelerateConfig

    return mapping


# Generate the mapping
ACCELERATE_CONFIG_MAPPING = get_accelerate_config_mapping()
