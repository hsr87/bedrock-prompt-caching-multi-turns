import time
import boto3
import json
import os
import pandas as pd
from anthropic import AnthropicBedrock
from functools import wraps
import random 

# Initialize AnthropicBedrock client
client = AnthropicBedrock(aws_region="us-west-2")

with open('RomeoAndJuliet.txt', 'r') as file: 
    sample_text = file.read()

result_dir = "37_250630_ttft"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    
n_experiments = 10
n_turns = 10

def retry_with_exponential_backoff(
    max_retries=5,
    initial_delay=2,
    exponential_base=2,
    jitter=True
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            num_retries = 0
            delay = initial_delay

            while True:
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    num_retries += 1
                    if num_retries > max_retries:
                        raise e

                    delay *= exponential_base
                    if jitter:
                        delay *= (0.5 + random.random())

                    time.sleep(delay)

        return wrapper
    return decorator

# 재시도 로직을 제거한 순수 API 호출 함수
def anthropic_bedrock_model_api_call(model_id, messages):
    start_time = time.time()
    ttft = None
    full_response = ""
    
    stream = client.messages.create(
        model=model_id,
        max_tokens=256,
        temperature=0.7,
        system=[
            {
                "type": "text", 
                "text": "You are a helpful assistant that answers questions concisely."
            }
        ],
        messages=messages,
        stream=True  # Streaming
    )
    for event in stream:
        if event.type == "content_block_start":
            # TTFT
            if ttft is None:
                ttft = time.time() - start_time
        
        elif event.type == "content_block_delta":
            # Txt
            if hasattr(event.delta, 'text'):
                full_response += event.delta.text
        
        elif event.type == "message_stop":
            # usage
            usage_data = getattr(event, 'amazon-bedrock-invocationMetrics')
    
    end_time = time.time()
    total_latency = end_time - start_time
    
    return full_response, usage_data, ttft, total_latency

# 재시도 로직을 적용한 래퍼 함수
@retry_with_exponential_backoff()
def anthropic_bedrock_model_with_ttft(model_id, messages):
    # 재시도 로직과 관계없이 실제 API 호출 시간만 측정
    return anthropic_bedrock_model_api_call(model_id, messages)
    
# Helper function to remove cache_control from a message
def remove_cache_control(message):
    if message["role"] == "user":
        new_content = []
        for item in message["content"]:
            if item.get("type") == "text" and "cache_control" in item:
                # Create a new item without cache_control
                new_item = {
                    "type": "text",
                    "text": item["text"]
                }
                new_content.append(new_item)
            else:
                new_content.append(item)
        message["content"] = new_content
    return message


for exp_num in range(n_experiments): 
    
    all_experiments_data = []
    print(f"Running experiment {exp_num+1}/{n_experiments}")
    
    # We'll use different questions for each turn to simulate a real conversation
    questions = [
        "Please summarize the storyline of the play.",
        "Who are the main characters in the tragedy?",
        "Why are the Montagues and Capulets in conflict with each other?",
        "What role does the Nurse play in Juliet's life?",
        "How does Romeo respond after killing Tybalt?",
        "What advice does Friar Lawrence give to Romeo after his banishment?",
        "Why does Paris visit the Capulet tomb in the final scene?",
        "What message fails to reach Romeo and what are the consequences?",
        "How do the parents react to finding their children dead?",
        "What reconciliation occurs between the families at the end of the play?"
    ]
    
    # Tracking metrics for this experiment
    experiment_data = []
    
    # Conversation history
    conversation = []
    
    # Keep track of which messages have cache_control
    cached_message_indices = []
    
    # Simulate n_turns
    for turn in range(n_turns):
        print(f"  Turn {turn+1}/{n_turns}: {questions[turn]}")

        if len(cached_message_indices) >= 4:
            oldest_cached_index = cached_message_indices.pop(0)
            conversation[oldest_cached_index] = remove_cache_control(conversation[oldest_cached_index])
        
        # Construct messages for this turn
        messages = []
    
        # Add the conversation history
        messages.extend(conversation)
    
        # Add the current question
        if turn == 0:
            current_message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": sample_text,
                    },
                    {
                        "type": "text",
                        "text": questions[turn] + " ",
                        "cache_control": {
                            "type": "ephemeral"  # Cache this content
                        }
                    },
                ]
            }
        else:
            current_message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": questions[turn] + " ",
                        "cache_control": {
                            "type": "ephemeral"  # Cache this content
                        }
                    },
                ]
            }
    
        messages.append(current_message)
    
        # Make the API call with TTFT measurement
        print(f"-------------{turn}-------------")
        print(messages)
        
        full_response, metrics, ttft, invocation_latency = anthropic_bedrock_model_with_ttft(
            model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            messages=messages,
        )
        
        # Update conversation history with message containing cache_control
        conversation.append(current_message)
        cached_message_indices.append(len(conversation)-1)
        
        # Add assistant response to conversation
        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": full_response,
                }
            ]
        })
    
        # Store data for this turn
        turn_data = {
            "experiment": exp_num + 1,
            "turn": turn + 1,
            "question": questions[turn],
            "input_tokens": metrics["inputTokenCount"],
            "output_tokens": metrics["outputTokenCount"],
            "cache_creation_input_tokens": metrics["cacheWriteInputTokenCount"] or 0,
            "cache_read_input_tokens": metrics["cacheReadInputTokenCount"] or 0,
            "invocation_latency": invocation_latency,
            "invocation_latency_bedrock": metrics["invocationLatency"],
            "first_byte_latency": metrics["firstByteLatency"],
            "ttft": ttft  
        }
        print(turn_data)
        experiment_data.append(turn_data)

    all_experiments_data.extend(experiment_data)

    # Convert to DataFrame and save
    pd.DataFrame(all_experiments_data).to_csv(f"{result_dir}/cache_experiment_results_cache_control_added_test_{exp_num}.csv", index=False)