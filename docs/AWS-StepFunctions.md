# AWS Step-Functions with Lambda - Step by Step Guide

## Step-1
- Login AWS Cosole
- Open Lambda Service
- Create function (lambda)
- ![alt text](/static/aws-step-functions/image-1.png)
- if 'Create a new role with basic Lambda permissions' not worked then use the next option called 'Use an existing role'
- Likewise, create all the lambda functions needed for the project/application
- Go to 'Code' tab in lambda function
- Write lambda_handler function and save this function (File -> Save)
- Test the lambda function by clicking on 'Test' button
- Now click on 'Deploy' to deploy this lambda file or save the changes in lambda function
- ![alt text](/static/aws-step-functions/image-2.png)
- The lambda handler code can be uploaded as a zip file or from AWS S3 location through 'Upload from' option
- See the lambda functions created for the application
- ![alt text](/static/aws-step-functions/image-3.png)
- Open these lambdas one after one and then copy their Function ARN value (see the below example)
- ![alt text](/static/aws-step-functions/image-4.png)
- this is needed to pointing out which lambda to execute in Step-Functin as per user inputs

## Step-2
- Now, 'Create role' for Step-Function
    - Open IAM service (IAM Dashboard)
    - Go to 'Roles'
    - Click on 'Create role'
    - ![alt text](/static/aws-step-functions/image-5.png)
    - Click on 'Next' for 'Add permissions'
    - ![alt text](/static/aws-step-functions/image-6.png)
    - Expand 'AWSLambdaRole' policy (see this and don't change anything)
    - Click on 'Next'
    - Write a 'Role name' under 'Role details' and keep others default
    - Click on 'Create role'
    - ![alt text](/static/aws-step-functions/image-7.png)

## Step-3
- Open 'Step Function' service
- Click on 'Create state machine' button
- Choose a template -> take default one that is 'Blank' template
- Click on 'Code' tab
- Write step-function code (see the example below); it is similar to model builder workflow in ArcGIS
- ![alt text](/static/aws-step-functions/image-8.png)
- here, 'Variable' key under 'Choices' is interacting with user input
- user inputs try to match with 'StringEquals' value
- if matches then it select 'Next' key and invoke the corresponding lambda function by finding its ARN value from 'Resource' key
- Now, click on 'Config' tab and give a 'State machine name'
- Select exisiting role for 'Execution role' under Permissions tab
- Keep others thing default
- Now, click on 'Create' button 
- ![alt text](/static/aws-step-functions/image-9.png)
- It will create a state-machine like below
- ![alt text](/static/aws-step-functions/image-10.png)

## Step-4
- Now, start execution/test the step-function
- It will open a window like below where we have to provide inputs with json format and we can write a name for the exution as well 
- when we click on 'Start execution' it will have following default values
- ![alt text](/static/aws-step-functions/image-11.png)
- after editing these defailt value it will be look like this as per our user inputs of the application
- ![alt text](/static/aws-step-functions/image-12.png)
- Now, click on 'Start execution'
- Results:
- ![alt text](/static/aws-step-functions/image-13.png)
- ![alt text](/static/aws-step-functions/image-14.png)
- we can view all the details of the process by clicking on each and every buttons on graph-view
- we can also view the logs in 'CloudWatch' by clicking on 'Log group' of the process
- ![alt text](/static/aws-step-functions/image-15.png)
- or,
- ![alt text](/static/aws-step-functions/image-16.png)
- Delete all the services in AWS that is created for demo purpose
- Further details can be found at [here](https://www.youtube.com/watch?v=s0XFX3WHg0w)
