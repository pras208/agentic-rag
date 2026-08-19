import boto3
import json
from typing import Generator

class BedrockClient:
    """Wrapper for AWS Bedrock models using the Converse API."""

    def __init__(self, region: str = "us-east-1", model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"):
        self.region = region
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def invoke(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
        """Invoke model and get response using the Converse API."""
        
        response = self.client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inference_config={"maxTokens": max_tokens, "temperature": temperature}
        )

        return response['output']['message']['content'][0]['text']

    def invoke_stream(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> Generator:
        """Stream model response token by token using the Converse API."""

        response = self.client.converse_stream(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inference_config={"maxTokens": max_tokens, "temperature": temperature}
        )

        for event in response['stream']:
            if 'contentBlockDelta' in event:
                yield event['contentBlockDelta']['delta']['text']