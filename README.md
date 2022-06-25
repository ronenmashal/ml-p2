# ml-p2
Machine Learning course, Project 2: training and using NN
# Requirements
1. Python 3.10
2. Recommended: vscode (or pycharm)

# Getting started
## Setup
1. Clone the repository to a local folder on your system.
2. Create python virtual environment in that folder:<br/>
   ```...\project-folder > python -m venv .venv```
3. Import the requirements:<br/>
   ```...\project-folder > pip install -r .\base-image\requirements.txt```
4. Start vscode in the folder:<br/>
   ```...\project-folder > code .```

## Training the model
5. Select the file site/train-main.py
6. Select the 'Run and Debug' view (left side bar):<br/>
  ![image](https://user-images.githubusercontent.com/28804769/175768154-ebbbe0e8-9fbe-4b90-8f16-2c2248d91932.png)
7. Make sure the selected configuration is 'Python: current file'
8. Hit F5 to start the execution.

## Teting the model
10. Test the model.
11. Run flask and test-app to see how it works.

# Project structure
- Root folder has the Dockerfile for creating the flask app container.
- Base-image folder has Dockerfile for building a base image with the required python packages.
- site folder has the py files for training and running models + flask app.
- test-app folder has an application to select an image from MNIST and send it to the web-service.

