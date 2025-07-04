import time
import json
import os
import pandas as pd
from anthropic import Anthropic
from functools import wraps
import random 

# Initialize Anthropic client
client = Anthropic(api_key="YOUR_API_KEY")

with open('RomeoAndJuliet.txt', 'r') as file: 
    sample_text = file.read()

result_dir = "37_250627_1p_ttft"
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

@retry_with_exponential_backoff()
def anthropic_model_with_ttft(model_id, messages):
    start_time = time.time()
    ttft = None
    full_response = ""
    usage_data = {}
    
    # 스트리밍으로 메시지 생성
    with client.messages.stream(
        model=model_id,
        max_tokens=256,
        temperature=0.7,
        system="You are a helpful assistant that answers questions concisely.",
        messages=messages,
    ) as stream:
        for event in stream:
            # content_block_start 이벤트에서 TTFT 측정
            if event.type == "content_block_start":
                if ttft is None:
                    ttft = time.time() - start_time
            
            # text_delta 이벤트에서 텍스트 수집
            elif event.type == "content_block_delta":
                if hasattr(event.delta, 'text'):
                    full_response += event.delta.text
            
            # message_delta 이벤트에서 usage 정보 수집
            elif event.type == "message_delta":
                if hasattr(event, 'usage'):
                    usage_data = event.usage
        
        # 스트림이 끝나면 최종 메시지에서 전체 usage 정보 가져오기
        final_message = stream.get_final_message()
        if hasattr(final_message, 'usage'):
            usage_data = {
                "input_tokens": final_message.usage.input_tokens,
                "output_tokens": final_message.usage.output_tokens,
                "cache_creation_input_tokens": getattr(final_message.usage, 'cache_creation_input_tokens', 0),
                "cache_read_input_tokens": getattr(final_message.usage, 'cache_read_input_tokens', 0)
            }
    
    end_time = time.time()
    total_latency = end_time - start_time
    
    return full_response, usage_data, ttft, total_latency
    
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
        
        full_response, metrics, ttft, invocation_latency = anthropic_model_with_ttft(
            model_id="claude-3-7-sonnet-20250219",  # 또는 "claude-3-haiku-20240307" 등
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
            "input_tokens": metrics.get("input_tokens", 0),
            "output_tokens": metrics.get("output_tokens", 0),
            "cache_creation_input_tokens": metrics.get("cache_creation_input_tokens", 0),
            "cache_read_input_tokens": metrics.get("cache_read_input_tokens", 0),
            "invocation_latency": invocation_latency,
            "ttft": ttft  
        }
        print(turn_data)
        experiment_data.append(turn_data)

    all_experiments_data.extend(experiment_data)

    # Convert to DataFrame and save
    pd.DataFrame(all_experiments_data).to_csv(f"{result_dir}/cache_experiment_results_cache_control_added_test_{exp_num}.csv", index=False)