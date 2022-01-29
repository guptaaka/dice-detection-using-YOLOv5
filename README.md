How to train the model that detects classes of dice/coins in a given image, using [YOLOv5](https://github.com/ultralytics/yolov5):

The model can be trained either on your own/custom machine or the NEU discovery cluster. The latter one requires a Northeastern University account. (Docs: https://rc-docs.northeastern.edu/en/latest/welcome/welcome.html). Would highly recommend using a GPU for this task, as training takes quite a bit of resources and time to complete. It took ~4 hours on a p100 GPU (https://rc-docs.northeastern.edu/en/latest/using-discovery/workingwithgpu.html).

### If you're using your custom machine (which is not the NEU Discovery cluster), you can stop reading here and follow the steps on [train_on_own_machine.md](https://github.com/guptaaka/coin-detection/blob/master/train_on_own_machine.md).

# Steps for running on Northeastern Univeristy Discovery cluster

Connect to the Discovery cluster: ssh <NEU-username>@login.discovery.neu.edu

Run **Step I** and **Step II** from the [train_on_own_machine.md](https://github.com/guptaaka/coin-detection/blob/master/train_on_own_machine.md) file, to get the dataset ready.

  Example image (The model would detect the labels and bouding boxes for the two dice/coins in the images):
  
  ![image](https://user-images.githubusercontent.com/23294197/149733638-bbc43262-76d3-4f7c-877b-e0fa0f8c5411.jpeg)
  ![image](https://user-images.githubusercontent.com/23294197/149734324-0a5a2049-b19c-406c-b8c7-6097c84264db.png)


**Step III. [Optional, only to test if you can really get connected to a GPU] Create the PyTorch environment by running below commands.**

```
srun --partition=gpu --nodes=1 --pty --export=All --gres=gpu:1 --mem=4G --time=01:00:00 /bin/bash 
module load cuda/11.3
module load anaconda3/2021.05
conda create --name pytorch_env_new python=3.7
conda activate pytorch_env_new
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
#(You can also install specific version if you like, conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch, instructions from https://pytorch.org/get-started/previous-versions/)
python -c'import torch; print(torch.cuda.is_available())'
#(This print statement should return "True". If it does not, it implies you were not able to connect to a GPU. You can still proceed, but the commands would be slow as they would instead run on a CPU.)
```

**Step IV. Create a batch file named _runModel.batch_, or any other name you like, with below content.**

```
#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --job-name=yolov5_65
#SBATCH --mem=3G
#SBATCH --export=All
#SBATCH --gres=gpu:p100:1
#SBATCH --partition=gpu
#SBATCH --output=output_65epochs/exec.%j.out
module load cuda/11.3
module load anaconda3/2021.05
source /shared/centos7/anaconda3/2021.05/etc/profile.d/conda.sh
conda activate pytorch_env_new
pip install -qr requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
echo $CONDA_DEFAULT_ENV
echo $(python -c'import torch; print(torch.cuda.is_available());')
echo $(python -c 'import torch; print(torch.cuda.current_device())')
echo $(python -c'import torch; print(torch.cuda.get_device_properties(torch.cuda.current_device()));')
echo $(python -c'import torch; print(min((int(arch.split("_")[1]) for arch in torch.cuda.get_arch_list()), default=35))')
time python train.py --epochs 65 --data ./dicedataset/data.yaml --cfg ./models/custom_yolov5s.yaml --weights '' --name yolov5s_results_initial --batch-size=12
```
  
**Step V. Run the batch file to start training the model. Note the job ID after running the command below.**
```
mkdir output_65epochs # to store logs
sbatch runModel.batch
```
**Step VI. Wait until the batch runs completely. You can run the below commands to check the job status,**
```
seff <jobID>
squeue -u <username>
```
or, you can also check the output from this batch file. For instance, if you want to check the output of your print statements from the batch file.
```
less</cat/tail> output_65epochs/exec.<jobID>.out
```
**Step V. Test.**

Test can be run either on CPU or GPU. I ran on the GPU by following the same steps as in Step IV above, but this time with the command below. It takes about 2-3 minutes to finish, so you can request a different GPU for less amount of time. 

```
time python detect.py --weights runs/train/yolov5s_results_initial/weights/best.pt --conf-thres 0.6 --source ../test/images --save-txt --save-conf
```
Open the log file and check the directory path where results are saved. You are looking for a line similar to this one, "2553 labels saved to runs/detect/exp7/labels". Edit the script *eval.py* with the correct directory path and run it. This script prints out the number of images that were detected correctly in entirety.

A huge thanks to Dr Ravi Sundaram and Khoury College at Northeastern University!
