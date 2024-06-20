from dotenv import load_dotenv
import boto3
import os
from botocore.exceptions import ClientError
import logging

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
    service_name="bedrock-agent",region_name=region_name)

# Define request headers, for Amazon Bedrock Model invocations
accept = 'application/json'
contentType = 'application/json'

#function that fetches knowledge bases from bedrock api
def fetch_knowledge_bases():
    try:
        response = client.list_knowledge_bases()

        knowledge_base_summaries = response['knowledgeBaseSummaries']
    except ClientError as e:
        print(f"Couldn't list knowledge bases: {e}")
        raise
    else:
        return knowledge_base_summaries
    
def get_knowledge_base(knowledgeBaseId):
    try:
        response = client.get_knowledge_base(knowledgeBaseId=knowledgeBaseId)

        knowledge_base = response['knowledgeBase']
    except ClientError as e:
        print(f"Couldn't get knowledge base: {e}")
        raise
    else:
        return knowledge_base