# Intuitive_RS3_slip_tutorial
Code base for the Intuitive RS3 Slip Detection Tutorial in Robotics

# Requirements
Ubuntu 18.04/20.04
 - Python == 3.6.9 

Windows 10 
 - Python == 3.7.7
 - Microsoft Visual Studio Build Tools, follow the tutorial from #3 - https://deepakjogi.medium.com/how-to-install-pybullet-physics-simulation-in-windows-e1f16baa26f6
 
MacOS
- Python == 3.7.7

# Installation
- Install PyCharm Community Edition IDE
- Setup virtual environment by following the guide - https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html#python_create_virtual_env
- In the terminal of the Pycharm execute the following
  - `source venv/bin/activate`
  - `pip install -r requirements_<insert_your_os_version(ubunut/macos/windows)>.txt`
  - `python test_installation.py`

# Codes
-`test_installation.py` - Run this to test your installation. In case of errors, please contact anirvan.dutta@bmwgroup.com or raise issue in Github \
-`tutorial_sim_slip_analysis.py` - Part-I tutorial for running the data-collection pipeline \
-`tutorial_sim_slip_detection.py` - Part II tutorial for training the DL model

# Dataset
- Download the 'slip_database.pickle' from [here](https://drive.google.com/file/d/1lubQyeWzDzlElCUEZJBHMsODiSh2vNai/view?usp=share_link)
