# Value Summation MBRL

Code for Value Summation: A Novel Scoring Function for MPC-based Model-based Reinforcement Learning (https://arxiv.org/pdf/2209.08169.pdf)

In this repository we provide code for Val-Sum algorithm described in the paper linked above. If you find this repository useful for your research, please cite:

```
@article{raisi2022value,
  title={Value Summation: A Novel Scoring Function for MPC-based Model-based Reinforcement Learning},
  author={Raisi, Mehran and Noohian, Amirhossein and Mccutcheon, Luc and Fallah, Saber},
  journal={arXiv preprint arXiv:2209.08169},
  year={2022}
  url          = {https://arxiv.org/pdf/2209.08169.pdf},
}
```
To run experiments in the paper, you will have to specify the environment names from these options: 'InvertedDoublePendulum-v2', 'Hopper-v3', 'Humanoid-v3', 'HumanoidStandup-v2', 'Pusher-v2', 'Reacher-v2', 'Ant-v3', 'HalfCheetah-v3'.
It is recommended to install gym version '0.19.0'. For instance you can use the following command to train the agent:
```
python main.py --env_id InvertedDoublePendulum-v2 --instance_number 1
```
In case of any questions, bugs, suggestions or improvements, please feel free to contact me at m.raisi@surrey.ac.uk  or open an issue.

