# CNN Classifier Project

### Workflow

    1. reserach
    2. config.yaml
    3. params.yaml
    4. config_entity.py
    5. configuration.py
    6. component
    7. pipeline
    8. training
    9. prediction
    10. app.py for web application
    11. main.py for run the pipeline
    12. dvc.yaml for the reprodcuing the pipeline

### Project structure

<pre>
CNN-Classifier
├───.dvc
├───.github
│   └───workflows
├───.vscode
├───artifacts
│   ├───data_ingestion
│   │   └───cifar-100-python
│   └───predict
├───configs
├───logs
├───research
├───src
│   └───CNNClassifier
│       ├───components
│       │   └───__init__.py
│       ├───config
│       │   └───__init__.py
│       ├───constants
│       │   └───__init__.py
│       ├───entity
│       │   └───__init__.py
│       ├───pipeline
│       │   └───__init__.py
│       ├───utils
│       │   └───__init__.py
│       └───__init__.py
├───static
│   └───image
├───templates
└───tests
    ├───integration
    └───unit
</pre>

### Run this application:

    Step-1: Clone the repository <repo link>
    Step-2: Create anaconda/miniconda environment after opening the repository
        conda create -n myenv python=3.10 -y
        conda activate myenv
    Step-3: install the requirements
        pip install -r requirements.txt
    Step-4: run app
        python app.py
    Step-5: Open up your local host and port

### Basic DVC Workflow:

    1. DVC Setup:
        $ pip install dvc
            - it will install dvc
        $ dvc init
            - it will create a .dvc folder that holds configuration information
        $ dvc add . /data_given/sample.csv
            - mapping remote storage on local machine
    2. DVC pipeline (data loading, pre-processing, data splitting, model building, metric measurement, etc.)
        - Create a dvc.yaml file where we need to define all the stages of the pipeline.
            $ dvc dag
        - Run the pipeline to see if all the stages are executed sequentially as defined:
            $ dvc repro
        - After successfully execuation, dvc will create a file called dvc.lock which tracks all the changes.
    3. Visualize model metrics
        $ dvc metrics show
    4. Push the code to github before proceeding further
        $ git add . && git commit -m "experiment with dvc" && git push origin main
    5. Plot and see confusion matrix
        $ dvc plots show cm.csv --template confusion -x chd -y Predicted
    6. Compare two different experiment with DVC
        $ dvc metrics diff
        $ dvc plots show cm.csv --template confusion -x chd -y Predicted


    # For more details: https://dvc.org/

    # Usage of Git and DVC:
        - Git is used to store and version code
        - DVC does the same for data and model files

### AWS CI/CD Deployment with Github Actions:

    1.  Login to AWS console
    2.  Create IAM user for deployment

        # with specific access

            - EC2 access : It is a virtual machine
            - ECR: Elastic Container Registry to save your docker image in AWS

        # Description: About the deployment

            - Build docker image of the source code
            - Push your docker image to ECR
            - Launch Your EC2
            - Pull Your image from ECR in EC2
            - Lauch your docker image in EC2

        # Policy

            - AmazonEC2ContainerRegistryFullAccess
            - AmazonEC2FullAccess

    3.  Create ECR repo to store/save docker image

        - Save the URI

    4.  Create EC2 machine (Ubuntu)

    5.  Open EC2 and Install docker in EC2 Machine:

        #optinal

            - sudo apt-get update -y
            - sudo apt-get upgrade

        #required

            - curl -fsSL https://get.docker.com -o get-docker.sh
            - sudo sh get-docker.sh
            - sudo usermod -aG docker ubuntu
            - newgrp docker

    6.  Configure EC2 as self-hosted runner:

        - setting>actions>runner>new self hosted runner> choose os> then run command one by one

    7.  Setup github secrets:
        - AWS_ACCESS_KEY_ID
        - AWS_SECRET_ACCESS_KEY
        - AWS_REGION
        - AWS_ECR_LOGIN_URI
        - ECR_REPOSITORY_NAME

### AZURE CI/CD Deployment with Github Actions

    - Save pass
    - Run from terminal
        -> docker build -t chickenapp.azurecr.io/chicken:latest .
        -> docker login chickenapp.azurecr.io
        -> docker push chickenapp.azurecr.io/chicken:latest
    - Deployment Steps:
        1. Build the Docker image of the Source Code
        2. Push the Docker image to Container Registry
        3. Launch the Web App Server in Azure
        4. Pull the Docker image from the container registry to Web App server and run

#### The pipeline contains the following sequential steps:

- config.yaml/params.yaml -> config_entity.py -> configuration.py -> components -> pipeline -> main.py -> flask app.py

#### Notes:

- get project structure => tree /f
