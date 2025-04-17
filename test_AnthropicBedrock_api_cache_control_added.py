import time
import boto3
import json
import os
import pandas as pd
from anthropic import AnthropicBedrock

# Initialize Anthropic client
client = AnthropicBedrock()

with open('RomeoAndJuliet.txt', 'r') as file: 
    sample_text = file.read()

all_experiments_data = []
    
n_experiments = 1
n_turns = 10
    
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
    
        # Make the API call
        start_time = time.time()
        response = client.messages.create(
            model="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            max_tokens=256,
            temperature=0.7,
            system=[
                {
                    "type": "text", 
                    "text": "You are a helpful assistant that answers questions concisely."
                }
            ],
            messages=messages
        )
        end_time = time.time()
        invocation_latency = end_time - start_time
        
        # Update conversation history with message containing cache_control
        conversation.append(current_message)
        cached_message_indices.append(len(conversation)-1)
        
        # Add assistant response to conversation
        conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": response.content[0].text
                }
            ]
        })
    
        # Get metrics
        metrics = response.usage
    
        # Store data for this turn
        turn_data = {
            "experiment": exp_num + 1,
            "turn": turn + 1,
            "question": questions[turn],
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "cache_creation_input_tokens": metrics.cache_creation_input_tokens or 0,
            "cache_read_input_tokens": metrics.cache_read_input_tokens or 0,
            "invocation_latency": invocation_latency
        }
        print(turn_data)
        experiment_data.append(turn_data)
        time.sleep(70)
    
    all_experiments_data.extend(experiment_data)


# Convert to DataFrame and save
pd.DataFrame(all_experiments_data).to_csv("cache_experiment_results_cache_control_added.csv", index=False)