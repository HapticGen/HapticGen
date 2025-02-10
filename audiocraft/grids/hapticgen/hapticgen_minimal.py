from ..musicgen._explorers import LMExplorer
from ...environment import AudioCraftEnvironment

from dora.explore import Launcher

@LMExplorer
def explorer(launcher: Launcher):
    # partitions = AudioCraftEnvironment.get_slurm_partitions(['team', 'global'])

    MIG_NODES="sg048,sg049,sg050,sg002" # for example, delete if not needed

    # `dora grid hapticgen.hapticgen_minimal` needs to be run from a login node due to this slurm issue: https://harvardmed.atlassian.net/wiki/spaces/O2/pages/1586793613/Troubleshooting+Slurm+Jobs#Jobs-fail-with-the-message:-Unable-to-satisfy-CPU-bind-request
    launcher.slurm_(gpus=4, cpus_per_task=2, mem_per_gpu=80, partition="htc", exclude=f"{MIG_NODES}", constraint="a100_80|h100", time=240, setup=[ # setup should go in config/teams/x.yaml probably
        "module load cuda-11.8.0-gcc-12.1.0",
        "module load mamba/latest",
        "source activate audiocraftgpu"
    ])

    launcher.bind_(solver='audiogen/audiogen_haptic_base_8khz')

    fsdp = {'autocast': False, 'fsdp.use': True}
    medium = {'model/lm/model_scale': 'medium'}

    launcher.bind_(fsdp)
    launcher.bind_(compression_model_checkpoint="//reference/51eabea7/checkpoint.th")
    launcher.bind_(dset='haptic/haptic_8khz_wc_filtered.yaml') # use your haptic converted dataset here
    launcher.bind_({'dataset.metadata_suffixes': [".llama3-ahr.json", ".json"]})
    launcher(medium, {'optim.epochs': 20}) # 12c0dcd3


    launcher_wcf_valid = launcher.slurm(time=30).bind({'dset': 'haptic/haptic_8khz_wc_filtered.yaml', 'dataset.metadata_suffixes': [".uf1.json", ".llama3-ahr.json", ".json"], 'optim.negative_sample_weight': 0.1, "execute_only": "valid"})

    launcher_dpo = launcher.bind(dset='haptic/haptic_8khz_dpo1.6.yaml')
    launcher_dpo.bind_({
        'dataset.metadata_suffixes': [".dpo1.json", ".uf1.json", ".llama3-ahr.json", ".json"],
        'dataset.train.aug_p': 0.0,
        'dataset.batch_size': 64, # half the original batch size, because DPO dataset has YW and YL pairs. I should also half the learning rate, but since it was just from me guessing anyway, I'll just keep it the same
        'dataset.train.batch_size': 64,
        'optim.negative_sample_weight': 0.1,
        'continue_from': "//sig/12c0dcd3",
        'reference_model_statedict': f"{YOUR_REF_PATH}/reference/51eabea7_12c0dcd3/model/state_dict.bin" # ${DORA_REFERENCE}/12c0dcd3/checkpoint.th might also work, but this might need to be expanded
    })
    launcher_dpo(medium, { 'optim.epochs': 4, 'optim.lr': 1e-7, 'optim.dpo.ce_mix_beta': 0.01 }) # d684b3a7
    launcher_wcf_valid(medium, {"continue_from": "//sig/d684b3a7"}) # 423c1017