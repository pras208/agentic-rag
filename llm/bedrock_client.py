import boto3
import json
from typing import Generator

class BedrockClient:
    """Wrapper for AWS Bedrock Claude model."""

    def __init__(self, region: str = "us-east-1", model_id: str = "anthropic.claude-3-5-sonnet-20241022"):
        self.region = region
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def invoke(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
        """Invoke Claude and get response."""
        payload = {
            "anthropic_version": "bedrock-2023-06-01",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(payload)
        )

        result = json.loads(response["body"].read())
        return result["content"][0]["text"]

    def invoke_stream(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> Generator:
        """Stream Claude response token by token."""
        payload = {
            "anthropic_version": "bedrock-2023-06-01",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        response = self.client.invoke_model_with_response_stream(
            modelId=self.model_id,
            body=json.dumps(payload)
        )

        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk["type"] == "content_block_delta":
                yield chunk["delta"]["text"]
