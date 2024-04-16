from pypdf import PdfReader
from dotenv import load_dotenv
import boto3
import json
import os
from botocore.exceptions import ClientError
import logging
import streamlit as st

# Setting up a logger with default settings
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Loading environment variables from a .env file
load_dotenv()

if os.getenv("region_name") is None:
    region_name = 'us-east-1'
else:
    region_name = os.getenv("region_name")
    
# Setting up the default boto3 session with a specified AWS profile name
boto3.setup_default_session(profile_name=os.getenv("profile_name"))

# Instantiating the Amazon Bedrock Runtime Client

client = boto3.client(
    service_name="bedrock-runtime",region_name=region_name)

# Define request headers, for Amazon Bedrock Model invocations
accept = 'application/json'
contentType = 'application/json'


def text_extraction(pdf_path):
    """
    Extracts text from a PDF file.

    :param pdf_path: The path to the PDF file.
    :return: The extracted text from the PDF file as a string.
    """
    # Creating a PdfReader object to read the PDF file
    reader = PdfReader(pdf_path)
    # Initializing an empty string to store the extracted text
    text = ""
    # Looping through each page of the PDF and extracting text
    for page in reader.pages:
        # Extracting text from the current page and appending it to the text variable
        text += page.extract_text()
        # Adding a newline character after each page's text to separate them
        text += "\n"
    # Returning the concatenated text extracted from all pages of the PDF file
    if len(text) > 12000:
        st.warning("The extracted text from the PDF may be longer than some of the models input tokens. Proceed with Caution")
        print("extracted text from the PDF may be longer than some model's input token maximums... proceeding with caution ")

    return text


def invoke_anthropic(model_id, prompt="", prompt_context="", max_tokens="4096"):
    """
    Invokes an Anthropic model using Amazon Bedrock and the specified parameters.

    :param model_id: The ID of the Anthropic model to invoke.
    :param prompt: Optional. The default prompt highlighting the task the model is trying to perform, defined in the orchestrator.py file.
    :param prompt_context: The prompt context includes the extracted text from the PDF file.
    :param max_tokens: Optional. The maximum number of tokens to generate. Defaults to the value of the 'max_tokens' environment variable.
    :return: A tuple containing the generated output text, the number of input tokens used, and the number of output tokens generated.
    """
    # Print the model ID (for debugging purposes)
    # TODO: Do we want to take this out?
    
    if max_tokens is None:
        max_tokens = "4096"
    
    print(model_id)
    # If prompt_context is provided, prepend it to the prompt
    if prompt_context:
        prompt=f"Human: \n\n {prompt} \n\n <context>{prompt_context}</context> \n Assistant: \n\n"
    # Define the request body for invoking the Anthropic model, using the messages API structure
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ],
    }

    try:
        # Invoke the Anthropic model through Bedrock using the defined request body
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
        )
        # Extract information from the response
        result = json.loads(response.get("body").read())
        # Extract the input tokens from the response
        input_tokens = result["usage"]["input_tokens"]
        # Extract the output tokens from the response
        output_tokens = result["usage"]["output_tokens"]
        # Extract the output text from the response
        output_text = result["content"][0]["text"]
        # Return the output text, input tokens, and output tokens
        return output_text, input_tokens, output_tokens
    except ClientError as err:
        # Log and raise an error if invoking the model fails
        logger.error(
            "Couldn't invoke {model_id}. Here's why: %s: %s",
            err.response["Error"]["Code"],
            err.response["Error"]["Message"],
        )
        raise


def invoke_meta(model_id, prompt="", prompt_context="", max_tokens='4096'):
    """
    Invokes a Meta model using Amazon Bedrock and the specified parameters.

    :param model_id: The ID of the Meta model to invoke.
    :param prompt: Optional. The default prompt highlighting the task the model is trying to perform, defined in the orchestrator.py file.
    :param prompt_context: The prompt context includes the extracted text from the PDF file.
    :param max_tokens: Optional. The maximum number of tokens to generate. Defaults to the value of the 'max_tokens' environment variable.
    :return: A tuple containing the generated output text, the number of input tokens used, and the number of output tokens generated.
    """
    # Print the model ID (for debugging purposes)
    # TODO: Do we want to take this out?
    print(model_id)
    
    # If prompt_context is provided, prepend it to the prompt
    if prompt_context:
        prompt=f"{prompt} \n\n <context>{prompt_context}</context>"
    # Define the request body for invoking the Meta model
    request_body = json.dumps({"prompt": prompt,
                               "max_gen_len": max_tokens,
                               "temperature": 0.5,
                               "top_p": 0.5
                               })
    try:
        # Invoke the Meta model using the defined request body and headers
        response = client.invoke_model(
            modelId=model_id,
            body=request_body,
            accept=accept,
            contentType=contentType
        )
        # Extract information from the response
        response_body = json.loads(response.get('body').read())
        # Extract the input tokens from the response
        input_tokens = response_body['prompt_token_count']
        # Extract the output tokens from the response
        output_tokens = response_body['generation_token_count']
        # Extract the output text from the response
        output_text = response_body['generation']
        # Return the output text, input tokens, and output tokens
        return output_text, input_tokens, output_tokens
    except ClientError as err:
        # Log and raise an error if invoking the model fails
        logger.error(
            "Couldn't invoke {model_id}. Here's why: %s: %s",
            err.response["Error"]["Code"],
            err.response["Error"]["Message"],
        )
        raise


def invoke_mistral(model_id, prompt="", prompt_context="", max_tokens='4096'):
    """
        Invokes a Mistral model using Amazon Bedrock and the specified parameters.

        :param model_id: The ID of the Mistral model to invoke.
        :param prompt: Optional. The default prompt highlighting the task the model is trying to perform, defined in the orchestrator.py file.
        :param prompt_context: The prompt context includes the extracted text from the PDF file.
        :param max_tokens: Optional. The maximum number of tokens to generate. Defaults to the value of the 'max_tokens' environment variable.
        :return: A tuple containing the generated output text, the number of input tokens used, and the number of output tokens generated.
        """
    # Print the model ID (for debugging purposes)
    # TODO: Do we want to take this out?
    print(model_id)
    # If prompt_context is provided, prepend it to the prompt
    if prompt_context:
        prompt=f"{prompt} \n\n <context>{prompt_context}</context>"
    # Define the request body for invoking the Mistral model
    request_body = json.dumps({"prompt": prompt,
                               "max_tokens": max_tokens,
                               "temperature": 0,
                               "top_k": 200,
                               "top_p": 0.5
                               })
    try:
        # Invoke the Mistral model using the defined request body and headers
        response = client.invoke_model(
            modelId=model_id,
            body=request_body,
            accept=accept,
            contentType=contentType
        )
        # Extract information from the response
        response_body = json.loads(response.get('body').read())
        # Extract the input tokens from the response
        input_tokens = int(response['ResponseMetadata']['HTTPHeaders']['x-amzn-bedrock-input-token-count'])
        # Extract the output tokens from the response
        output_tokens = int(response['ResponseMetadata']['HTTPHeaders']['x-amzn-bedrock-output-token-count'])
        # Extract the output text from the response
        output_text = response_body['outputs'][0]['text']
        # Return the output text, input tokens, and output tokens
        return output_text, input_tokens, output_tokens
    except ClientError as err:
        # Log and raise an error if invoking the model fails
        logger.error(
            "Couldn't invoke {model_id}. Here's why: %s: %s",
            err.response["Error"]["Code"],
            err.response["Error"]["Message"],
        )
        raise


def invoke_cohere(model_id, prompt="", prompt_context="",  max_tokens='4096'):
    """
        Invokes a Cohere model using Amazon Bedrock and the specified parameters.

        :param model_id: The ID of the Cohere model to invoke.
        :param prompt: Optional. The default prompt highlighting the task the model is trying to perform, defined in the orchestrator.py file.
        :param prompt_context: The prompt context includes the extracted text from the PDF file.
        :param max_tokens: Optional. The maximum number of tokens to generate. Defaults to the value of the 'max_tokens' environment variable.
        :return: A tuple containing the generated output text, the number of input tokens used, and the number of output tokens generated.
        """
    # Print the model ID (for debugging purposes)
    # TODO: Do we want to take this out?
    print(model_id)
    # If prompt_context is provided, prepend it to the prompt
    if prompt_context:
        prompt=f"{prompt} \n\n <context>{prompt_context}</context>"
    # Define the request body for invoking the Cohere model
    request_body = json.dumps({"prompt": prompt,
                               "max_tokens": max_tokens,
                               "temperature": 0.5,
                               })
    try:
        # Invoke the Cohere model using the defined request body and headers
        response = client.invoke_model(
            modelId=model_id,
            body=request_body,
            accept=accept,
            contentType=contentType
        )
        # Extract information from the response
        response_body = json.loads(response.get('body').read())
        # Extract the input tokens from the response
        input_tokens = int(response['ResponseMetadata']['HTTPHeaders']['x-amzn-bedrock-input-token-count'])
        # Extract the output tokens from the response
        output_tokens = int(response['ResponseMetadata']['HTTPHeaders']['x-amzn-bedrock-output-token-count'])
        # Extract the output text from the response
        output_text = response_body['generations'][0]['text']
        # Return the output text, input tokens, and output tokens
        return output_text, input_tokens, output_tokens
    except ClientError as err:
        # Log and raise an error if invoking the model fails
        logger.error(
            "Couldn't invoke {model_id}. Here's why: %s: %s",
            err.response["Error"]["Code"],
            err.response["Error"]["Message"],
        )
        raise


def invoke_amazon(model_id, prompt="", prompt_context="", max_tokens='4096'):
    """
        Invokes an Amazon model using Amazon Bedrock and the specified parameters.

        :param model_id: The ID of the Amazon model to invoke.
        :param prompt: Optional. The default prompt highlighting the task the model is trying to perform, defined in the orchestrator.py file.
        :param prompt_context: The prompt context includes the extracted text from the PDF file.
        :param max_tokens: Optional. The maximum number of tokens to generate. Defaults to the value of the 'max_tokens' environment variable.
        :return: A tuple containing the generated output text, the number of input tokens used, and the number of output tokens generated.
        """
    # Print the model ID (for debugging purposes)
    # TODO: Do we want to take this out?
    print(model_id)
    # If prompt_context is provided, prepend it to the prompt
    if prompt_context:
        prompt=f"{prompt} \n\n <context>{prompt_context}</context>"
    # Define the request body for invoking the Amazon model
    request_body = json.dumps({"inputText": prompt,
                               "textGenerationConfig": {
                                   "maxTokenCount": max_tokens,
                                   "stopSequences": [],
                                   "temperature": 0.5,
                                   "topP": 0.5
                               }})
    try:
        # Invoke the Amazon model using the defined request body and headers
        response = client.invoke_model(
            modelId=model_id,
            body=request_body,
            accept=accept,
            contentType=contentType
        )
        # Extract information from the response
        response_body = json.loads(response.get('body').read())
        # Extract the input tokens from the response
        input_tokens = response_body['inputTextTokenCount']
        # Extract the output tokens from the response
        output_tokens = response_body['results'][0]['tokenCount']
        # Extract the output text from the response
        output_text = response_body['results'][0]['outputText']
        # Return the output text, input tokens, and output tokens
        return output_text, input_tokens, output_tokens
    except ClientError as err:
        # Log and raise an error if invoking the model fails
        logger.error(
            "Couldn't invoke {model_id}. Here's why: %s: %s",
            err.response["Error"]["Code"],
            err.response["Error"]["Message"],
        )
        raise


def invoke_AI21(model_id, prompt="", prompt_context="", max_tokens='4096'):
    """
        Invokes an AI21 model using Amazon Bedrock and the spescified parameters.

        :param model_id: The ID of the AI21 model to invoke.
        :param prompt: Optional. The default prompt highlighting the task the model is trying to perform, defined in the orchestrator.py file.
        :param prompt_context: The prompt context includes the extracted text from the PDF file.
        :param max_tokens: Optional. The maximum number of tokens to generate. Defaults to the value of the 'max_tokens' environment variable.
        :return: A tuple containing the generated output text, the number of input tokens used, and the number of output tokens generated.
        """
    # Print the model ID (for debugging purposes)
    # TODO: Do we want to take this out?
    print(model_id)
    # If prompt_context is provided, prepend it to the prompt
    if prompt_context:
        prompt=f"{prompt} \n\n <context>{prompt_context}</context>"
    # Define the request body for invoking the AI21 model
    request_body = json.dumps({"prompt": prompt,
                               "maxTokens": max_tokens,
                               "temperature": 0.5,
                               "topP": 0.5,
                               "stopSequences": [],
                               })
    try:
        # Invoke the AI21 model using the defined request body and headers
        response = client.invoke_model(
            modelId=model_id,
            body=request_body,
            accept=accept,
            contentType=contentType
        )
        # Extract information from the response
        response_body = json.loads(response.get('body').read())
        # Extract the input tokens from the response
        input_tokens = int(response['ResponseMetadata']['HTTPHeaders']['x-amzn-bedrock-input-token-count'])
        # Extract the output tokens from the response
        output_tokens = int(response['ResponseMetadata']['HTTPHeaders']['x-amzn-bedrock-output-token-count'])
        # Extract the output text from the response
        output_text = response_body['completions'][0]['data']['text']
        # Return the output text, input tokens, and output tokens
        return output_text, input_tokens, output_tokens
    except ClientError as err:
        # Log and raise an error if invoking the model fails
        logger.error(
            "Couldn't invoke {model_id}. Here's why: %s: %s",
            err.response["Error"]["Code"],
            err.response["Error"]["Message"],
        )
        raise
