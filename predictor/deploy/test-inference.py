import requests
import torch
import torch.nn.functional as F

def predict_clashroyale_with_torch(input_list: list) -> tuple:
    """
    Pads the input list, sends an inference request to the KServe model,
    applies softmax using PyTorch, and returns the top 5 predictions.

    Args:
        input_list (list): A list of integers representing the model input.

    Returns:
        tuple: A tuple containing two lists: the top 5 indices and their
               corresponding softmax probabilities.
    """
    # --- 1. Define constants and padding ---
    MAX_LENGTH = 7
    PADDING_IDX = 120
    INFERENCE_URL = "http://clashroyale.ddns.net/v2/models/clashroyale/infer"
    
    # Pad the input list to a length of 7
    if len(input_list) > MAX_LENGTH:
        print(f"Warning: Input list is longer than {MAX_LENGTH}. Truncating.")
        padded_list = input_list[:MAX_LENGTH]
    else:
        num_padding = MAX_LENGTH - len(input_list)
        padded_list = [PADDING_IDX] * num_padding + input_list
    
    print(f"Original input list: {input_list}")
    print(f"Padded input list:   {padded_list}")

    # --- 2. Construct the KServe V2 inference request body ---
    # The data needs to be a list of lists for the JSON body
    request_body = {
        "inputs": [
            {
                "name": "input-0",
                "shape": [1, MAX_LENGTH],
                "datatype": "INT64",
                "data": [padded_list]
            }
        ]
    }
    
    # --- 3. Send the HTTP POST request ---
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(INFERENCE_URL, json=request_body, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        response_data = response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"Error making inference request: {e}")
        return None, None
    
    # --- 4. Process the model's response with PyTorch ---
    if "outputs" not in response_data or not response_data["outputs"]:
        print("Error: 'outputs' key not found in the response or is empty.")
        return None, None
        
    # Extract the output list and convert it to a PyTorch tensor
    output_data = response_data["outputs"][0]["data"]
    output_tensor = torch.tensor(output_data, dtype=torch.float32)
    
    # --- 5. Apply the Softmax function using F.softmax ---
    # F.softmax expects the `dim` parameter. Since our tensor is 1D, dim=0 is correct.
    probabilities = F.softmax(output_tensor, dim=0)
    
    # --- 6. Get the top 5 predictions using torch.topk ---
    # `torch.topk` is a clean way to get both the values and indices of the top-k elements.
    top_k_values, top_k_indices = torch.topk(probabilities, k=5)
    
    # Convert the tensors to Python lists for a clean return value
    top_5_indices = top_k_indices.tolist()
    top_5_probabilities = top_k_values.tolist()
    
    return top_5_indices, top_5_probabilities

# --- Main execution block ---
if __name__ == "__main__":
    # Example input list
    input_to_predict = [21,10,109,85,107]
    
    top_indices, top_probabilities = predict_clashroyale_with_torch(input_to_predict)
    
    if top_indices is not None and top_probabilities is not None:
        print("\n--- Model Prediction Results ---")
        print("Top 5 predictions:")
        for i in range(5):
            # We use f-strings to format the output nicely
            print(f"  Rank {i + 1}: Index {top_indices[i]}, Probability {top_probabilities[i]:.4f}")