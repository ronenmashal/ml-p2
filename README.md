# ml-p2
Machine Learning course, Project 2: training and using NN
# Requirements
1. Python 3.10
1. Recommended: vscode (or pycharm)

# Getting started
## Setup
1. Clone the repository to a local folder on your system.
1. Create python virtual environment in that folder:<br/>
   ```...\project-folder > python -m venv .venv```
1. Import the requirements:<br/>
   ```...\project-folder > pip install -r .\base-image\requirements.txt```
1. Start vscode in the folder:<br/>
   ```...\project-folder > code .```

## Training the model
1. Select the file site/train-main.py
1. Select the 'Run and Debug' view (left side bar):<br/>
  ![image](https://user-images.githubusercontent.com/28804769/175768154-ebbbe0e8-9fbe-4b90-8f16-2c2248d91932.png)
1. Make sure the selected configuration is 'Python: current file'
1. Hit F5 to start the execution.

After the training is complete, you will see checkpoint data created in the designated folder:<br/>
![image](https://user-images.githubusercontent.com/28804769/175769566-f3158237-496b-4997-83f3-d49da8fb6adc.png)

## Teting the model
1. Select the file site/app.py
1. Update the file to use the latest training suffix (timestamp):<br/>
   ![image](https://user-images.githubusercontent.com/28804769/175769620-7e7eb694-0ff9-4294-8166-1591b80ba8f7.png)
3. Select the 'Run and Debug' view (left side bar)
4. Make sure the selected configuration is 'Python: Flask'<br/>
   ![image](https://user-images.githubusercontent.com/28804769/175769359-7612a6ee-29f6-4ea2-ae28-896ba4e526dd.png)
1. Hit F5 to start the execution.
   vscode will run the flask server.
1. Open a new CMD terminal in vscode, and run test-app:<br/>
   ![image](https://user-images.githubusercontent.com/28804769/175769726-bbbfed1f-0e04-423a-ba53-d87da8aa91e1.png)

When the application runs, you will see this application:
![image](https://user-images.githubusercontent.com/28804769/175769743-d969c82f-42dd-4627-bf8d-99a5a8d60312.png)
Now you can select an image and click 'check'. The results are printed to the console, from which you ran the application:<br/>
![image](https://user-images.githubusercontent.com/28804769/175769792-b589dc8e-ab21-4b9a-85d2-a8f167bae2fb.png)

# Project structure
- Root folder has the Dockerfile for creating the flask app container.
- Base-image folder has Dockerfile for building a base image with the required python packages.
- site folder has the py files for training and running models + flask app.
- test-app folder has an application to select an image from MNIST and send it to the web-service.

