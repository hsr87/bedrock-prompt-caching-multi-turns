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
n_turns = 6
    
# Questions for each turn
questions = [
    "Please summarize the story.",
    "What is the subject of the story?",
    "Where did Romeo and Juliet first meet?",
    "What is the name of the woman Romeo loved before?",
    "How does Mercutio die?",
    "What method did Juliet use to fake her death?",
]

for exp_num in range(n_experiments):
    print(f"Running experiment {exp_num+1}/{n_experiments}")
    experiment_data = []
    messages = []

    # Simulate n_turns
    for turn in range(n_turns):
        print(f"  Turn {turn+1}/{n_turns}: {questions[turn]}")

        start_time = time.time()
        
        if turn == 0:
            response_body = bedrock_runtime.converse(
                modelId='us.anthropic.claude-3-7-sonnet-20250219-v1:0',
                messages=[
                    {
                        'role': 'user',
                        'content': [
                            {
                                "text": sample_text,
                            },
                            {
                                "text": questions[turn] + " "
                            },
                            {
                                'cachePoint': {
                                    'type': 'default'  # Insert a cache point after the question
                                }
                            },
                        ]
                    }
                ],
                system=[
                    {
                        'text': 'You are a helpful assistant that answers questions concisely.'
                    }
                ],
                inferenceConfig={
                    'maxTokens': 256,
                    'temperature': 0.7,
                    'topP': 0.8
                }
            )
            
            # Initialize conversation history
            messages = [
                {
                    'role': 'user',
                    'content': [
                        {
                            "text": sample_text,
                        },
                        {
                            "text": questions[turn] + " "
                        }
                    ]
                },
                {
                    'role': 'assistant',
                    'content': [
                        {
                            "text": response_body['output']['message']['content'][0]['text']
                        }
                    ]
                }
            ]
            
        else:
            # Add the new question to messages
            messages.append(
                {
                    'role': 'user',
                    'content': [
                        {
                            "text": questions[turn] + " "
                        },
                        {
                            'cachePoint': {
                                'type': 'default'  # Insert a cache point after the question
                            }
                        },
                    ]
                }
            )
            
            response_body = bedrock_runtime.converse(
                modelId='us.anthropic.claude-3-7-sonnet-20250219-v1:0',
                messages=messages,
                system=[
                    {
                        'text': 'You are a helpful assistant that answers questions concisely.'
                    }
                ],
                inferenceConfig={
                    'maxTokens': 256,
                    'temperature': 0.7,
                    'topP': 0.8
                }
            )
            
            # Update the last message in history (remove cachePoint)
            messages[-1] = {
                'role': 'user',
                'content': [
                    {
                        "text": questions[turn] + " "
                    }
                ]
            }
            
            # Add assistant's response to conversation history
            messages.append({
                'role': 'assistant',
                'content': [
                    {
                        "text": response_body['output']['message']['content'][0]['text']
                    }
                ]
            })
        
        end_time = time.time()
        invocation_latency = end_time - start_time

        # Extract metrics from the response
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

        experiment_data.append(turn_data)
        time.sleep(60)

    all_experiments_data.extend(experiment_data)

# Convert to DataFrame
df = pd.DataFrame(all_experiments_data)

# Save raw data
df.to_csv("cache_experiment_results_converse.csv", index=False)