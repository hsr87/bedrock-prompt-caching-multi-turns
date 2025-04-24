import time
import boto3
import json
import os
import pandas as pd

# Initialize Bedrock runtime client
bedrock_runtime = boto3.client('bedrock-runtime')

with open('RomeoAndJuliet.txt', 'r') as file: 
    sample_text = file.read()

all_experiments_data = []
    
n_experiments = 1
n_turns = 10
    
# Helper function to remove cachePoint from a message
def remove_cache_point(message):
    if message["role"] == "user":
        new_content = [item for item in message["content"] if not (isinstance(item, dict) and 'cachePoint' in item)]
        message["content"] = new_content
    return message

for exp_num in range(n_experiments): 
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
    
    # Keep track of which messages have cachePoint
    cached_message_indices = []
    
    # Simulate n_turns
    for turn in range(n_turns):
        print(f"  Turn {turn+1}/{n_turns}: {questions[turn]}")

        if len(cached_message_indices) >= 4:
            oldest_cached_index = cached_message_indices.pop(0)
            conversation[oldest_cached_index] = remove_cache_point(conversation[oldest_cached_index])
        
        # Prepare messages for this API call
        messages = conversation.copy()
        
        # Add the current question with cache point
        if turn == 0:
            # First turn includes sample text
            current_message = {
                "role": "user",
                "content": [
                    {
                        "text": sample_text,
                    },
                    {
                        "text": questions[turn] + " "
                    },
                    {
                        'cachePoint': {
                            'type': 'default'
                        }
                    }
                ]
            }
        else:
            # Subsequent turns just include the question
            current_message = {
                "role": "user",
                "content": [
                    {
                        "text": questions[turn] + " "
                    },
                    {
                        'cachePoint': {
                            'type': 'default'
                        }
                    }
                ]
            }
        
        messages.append(current_message)
        
        # Make the API call
        start_time = time.time()
        print(f"-------------{turn}-------------")
        print(messages)
        
        response_body = bedrock_runtime.converse(
            modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            messages=messages,
            system=[
                {
                    "text": "You are a helpful assistant that answers questions concisely."
                }
            ],
            inferenceConfig={
                'maxTokens': 256,
                'temperature': 0.7,
                'topP': 0.8
            }
        )
        
        end_time = time.time()
        invocation_latency = end_time - start_time
        
        # Add current message to conversation history (with cache point)
        conversation.append(current_message)
        cached_message_indices.append(len(conversation)-1)
        
        # Add assistant response to conversation history
        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "text": response_body['output']['message']['content'][0]['text']
                }
            ]
        })
    
        # Get metrics
        metrics = response_body['usage']
    
        # Store data for this turn
        turn_data = {
            "experiment": exp_num + 1,
            "turn": turn + 1,
            "question": questions[turn],
            "input_tokens": metrics['inputTokens'],
            "output_tokens": metrics['outputTokens'],
            "cache_creation_input_tokens": metrics.get('cacheWriteInputTokens', 0),
            "cache_read_input_tokens": metrics.get('cacheReadInputTokens', 0),
            "invocation_latency": invocation_latency
        }
        
        print(turn_data)
        experiment_data.append(turn_data)
        time.sleep(60)
    
    all_experiments_data.extend(experiment_data)

# Convert to DataFrame and save
pd.DataFrame(all_experiments_data).to_csv("cache_experiment_results_converse_api_3.csv", index=False)