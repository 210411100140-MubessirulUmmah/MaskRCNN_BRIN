@echo off
pip install torch==2.3.1
REM Clone the detectron2 repository
git clone https://github.com/facebookresearch/detectron2.git

REM Navigate to the detectron2 directory
cd detectron2

REM Install the requirements for detectron2
pip install -r requirements.txt

REM Install detectron2
pip install .
