
## Deploy and lambda

Training a linear regression model for predictive maintenance is an important step, but taking it into production can be challenging. Traditional deployment methods, like using a web server or Docker, often require a lot of work in setting up and maintaining infrastructure, leading to increased costs in development, hardware, and software management.

In this article, we will introduce you to AWS Lambda, a cloud service that allows you to run your code without worrying about the underlying infrastructure. By deploying a linear regression model trainer and predictor on AWS Lambda, you can save both time and money, while focusing on developing and improving your predictive maintenance models. This serverless approach simplifies the process for machine learning developers who may not be experts in cloud computing, allowing you to deliver accurate predictions and better maintain your physical systems without the burden of managing the infrastructure.

```text
1. install sam
2. create dev virtual env
3. sam init:

Which template source would you like to use?
        1 - AWS Quick Start Templates
        2 - Custom Template Location
Choice: 1

Choose an AWS Quick Start application template
        1 - Hello World Example
        2 - Multi-step workflow
        3 - Serverless API
        4 - Scheduled task
        5 - Standalone function
        6 - Data processing
        7 - Hello World Example With Powertools
        8 - Infrastructure event management
        9 - Serverless Connector Hello World Example
        10 - Multi-step workflow with Connectors
        11 - Lambda EFS example
        12 - DynamoDB Example
        13 - Machine Learning
Template: 1

Use the most popular runtime and package type? (Python and zip) [y/N]: y

Would you like to enable X-Ray tracing on the function(s) in your application?  [y/N]: N

Would you like to enable monitoring using CloudWatch Application Insights?
For more info, please view https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/cloudwatch-application-insights.html [y/N]: N

Project name [sam-app]: linreg-sam
```
