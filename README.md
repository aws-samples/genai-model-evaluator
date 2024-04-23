# GenAI Model Evaluator

The GenAI Model Evaluator is a tool designed for you to analyze and compare the performance of various Bedrock FM models, particularly focusing on aspects like cost efficiency, speed, and summarization accuracy of text from uploaded PDF documents. By automating the evaluation process and providing detailed scoring across multiple criteria, it enables you to make informed decisions when selecting the most optimal models for specific tasks. With its streamlite interface, you can easily upload your PDFs, run evaluations on different models, and visualize comparative performance metrics to identify the best model for your needs.

##### Authors: Brian Maguire, Dom Bavaro, Ryan Doty

## Demo
![Alt text](images/demo.gif)


## Features

- **PDF Summarization:** Currently supports evaluating Bedrock Models' ability to summarize text from uploaded PDF documents. Future updates will extend support to other tasks such as classification and text generation/editing.
- **Cost Efficiency and Speed Analysis:** Enables comparison of different FM models based on their total operational costs and execution time to pinpoint the most efficient options.
- **Summarization Evaluation with Detailed Scoring:** Offers functions that summarize evaluation results with scores reflecting various aspects like fidelity, coherence, conciseness, objectivity, etc.
- **Visualization Tools for Model Comparison:** Provides visual aids to facilitate an easier understanding of how different models stack up against each other in performance metrics.
- **Automated Model Evaluation:** Streamlines the evaluation process through AI-driven analysis of model performance data.

## Benefits

- **Efficiency:** Saves time and effort by automating analysis tasks.
- **Consistency:** Ensures consistent assessments across different models using standardized criteria.
- **Insights:** Facilitates informed decision-making by quantifying performance metrics.
- **Presentation:** Enhances understanding through clear visual comparisons between models.
- **Scalability:** Easily scales to accommodate multiple models or datasets.

## Getting Started

To begin using the GenAI Model Evaluator:

1. Ensure you have Amazon Bedrock Access and CLI Credentials.
2. Install Python 3.9 or 3.10 on your machine.
3. Clone this repository to your local environment.
4. Navigate to the project directory.
5. Optional: Set the .env settings
6. Set up a Python virtual environment and install required dependencies

### Configuration

Configure necessary environment variables (e.g., AWS credentials, database connections) as detailed in sample directories.

# How to use this Repo:

## Prerequisites:

1. Amazon Bedrock Access and CLI Credentials.
2. Ensure Python 3.10 installed on your machine, it is the most stable version of Python for the packages we will be using, it can be downloaded [here](https://www.python.org/downloads/release/python-3911/).

## Step 1:

The first step of utilizing this repo is performing a git clone of the repository.

```
git clone https://github.com/aws-samples/genai-model-evaluator.git

```


## Step 2:

Set up a python virtual environment in the root directory of the repository and ensure that you are using Python 3.9. This can be done by running the following commands:

```
pip install virtualenv
python3.10 -m venv venv

```

The virtual environment will be extremely useful when you begin installing the requirements. If you need more clarification on the creation of the virtual environment please refer to this [blog](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/).
After the virtual environment is created, ensure that it is activated, following the activation steps of the virtual environment tool you are using. Likely:

```
cd venv
cd bin
source activate
cd ../../

```

After your virtual environment has been created and activated, you can install all the requirements found in the requirements.txt file by running this command in the root of this repos directory in your terminal:

```
pip install -r requirements.txt

```
## Step 3:

Create a .env file in the root of this repo. Within the .env file you just created you will need to configure the .env to contain:

```
max_tokens=2048
region_name=us-east-1
profile_name=<AWS_CLI_PROFILE_NAME>
save_folder=<PATH_TO_ROOT_OF_THIS_REPO>

```


## Step 4:
### Running the Application

Run the Streamlit application using the command provided in each sample's directory for an interactive evaluation experience.

```
streamlit run app.py

```

## How the Evaluation Works

The Model Evaluator leverages an automated approach to assess the performance of AI models, utilizing `anthropic.claude-3-sonnet` as the evaluating model. This section explains the methodology, scale, and criteria used for evaluation.

### Evaluation Methodology

1. **Automated Analysis:** The evaluation process is automated leveraging the `anthropic.claude-3-sonnet` model with Amazon Bedrock. This model analyzes performance data of other models based on predefined criteria.

2. **Data Preparation:** The evaluator processes CSV data containing performance metrics such as 'Total Cost', 'Time Length', and 'Summary Score' of each model being evaluated.

3. **Criteria-Based Scoring:** Each model is scored based on specific evaluation criteria, including cost efficiency, speed, accuracy, completeness, logical flow, structure, conciseness, clarity, objectivity, tone consistency, and adherence to task instructions.

4. **Result Summarization:** The evaluation results are summarized to provide a comprehensive overview of each model's strengths and weaknesses across different metrics.

### Evaluation Scale

The scoring for each criterion is done on a scale of 0 to 5:

- **0 (Unacceptable):** Completely fails to meet the evaluation criterion.
- **1 (Poor):** Significantly underperforms against the evaluation criterion.
- **2 (Fair):** Shows some effort but falls short in meeting the criterion adequately.
- **3 (Good):** Meets the basic requirements of the evaluation criterion.
- **4 (Very Good):** Exceeds expectations with minor areas for improvement.
- **5 (Excellent):** Exceptionally meets or surpasses all aspects of the evaluation criterion.

### Summary Criterion Scoring

1. **Cost Efficiency:** Assesses how well a model manages operational costs relative to its output quality.
   
2. **Speed:** Evaluates the time taken by a model to perform tasks compared to others.
   
3. **Accuracy:** Measures how accurately a model's output matches expected or ground truth data.
   
4. **Completeness:** Checks if all relevant information is captured in a model's output.
   
5. **Logical Flow:** Looks at how logically coherent and structured a summary or response generated by a model is.
   
6. **Structure:** Examines paragraph and sentence structure within generated summaries for readability and organization.
   
7. **Conciseness:** Evaluates if models can convey information effectively in fewer words without losing essential content.
   
8. **Clarity:** Assesses how easily understandable and clear a model's outputs are to its intended audience.
   
9. **Objectivity:** Measures if outputs remain neutral and unbiased when summarizing or providing information based on input data.

10.  **Tone Consistency:** Ensures that models maintain consistency with the tone set by their input prompts or source material.

11.  **Adherence to Task Instructions:** Checks whether models follow given prompt instructions accurately without deviating from requested output format or content boundaries.

## Prerequisites
For detailed prerequisites, refer [here](https://github.com/aws-samples/genai-quickstart-pocs#prerequisites).

## Security
For security concerns or contributions, see [CONTRIBUTING](https://github.com/aws-samples/genai-quickstart-pocs/blob/main/CONTRIBUTING.md#security-issue-notifications).

## License
This project is licensed under the MIT-0 License. For more details, see [LICENSE](https://github.com/aws-samples/genai-quickstart-pocs/blob/main/LICENSE).

