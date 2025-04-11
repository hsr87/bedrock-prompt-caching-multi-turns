import time
import boto3  # Added boto3 import
import json   # Added json import
import os
import pandas as pd
import anthropic
# Initialize Anthropic client

client = anthropic.Anthropic(
    api_key="YOUR_API_KEY",  
)

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
    conversation = []

    # Simulate n_turns
    for turn in range(n_turns):
        print(f"  Turn {turn+1}/{n_turns}: {questions[turn]}")

        # Construct message content for this turn
        content = []
        if turn == 0:
            content.append({"type": "text", "text": sample_text})
        
        # Add the current question with ephemeral flag for caching
        content.append({
            "type": "text",
            "text": questions[turn] + " ",
            "cache_control": {"type": "ephemeral"}  # Cache this content
        })
        
        # Construct full messages list with history + current message
        messages = conversation.copy()
        messages.append({"role": "user", "content": content})
        
        # Prepare version without cache control for conversation history
        content_for_saving = []
        if turn == 0:
            content_for_saving.append({"type": "text", "text": sample_text})
        content_for_saving.append({"type": "text", "text": questions[turn] + " "})

        # Make the API call
        start_time = time.time()
        response = client.messages.create(
            model="claude-3-7-sonnet-20250219",
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
        
        # Update conversation history
        conversation.append({"role": "user", "content": content_for_saving})
        conversation.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response.content[0].text}]
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

        experiment_data.append(turn_data)
        time.sleep(1)
        
    all_experiments_data.extend(experiment_data)

# Convert to DataFrame
df = pd.DataFrame(all_experiments_data)

# Save raw data
df.to_csv("cache_experiment_results_anthropic.csv", index=False)