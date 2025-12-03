import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split

def split_dataset(input_csv_path, train_ratio=0.8, random_state=42):
    """
    Splits a dataset_info.csv file into train and test sets.
    
    Args:
        input_csv_path (str): Path to the input dataset_info.csv file.
        train_ratio (float): Ratio of the training set (default: 0.8).
        random_state (int): Random seed for reproducibility (default: 42).
    """
    if not os.path.exists(input_csv_path):
        print(f"Error: File not found at {input_csv_path}")
        return

    try:
        # Read the CSV file
        df = pd.read_csv(input_csv_path)
        
        # Split the dataset
        train_df, test_df = train_test_split(
            df, 
            train_size=train_ratio, 
            random_state=random_state,
            shuffle=True
        )
        
        # Get directory of the input file
        output_dir = os.path.dirname(input_csv_path)
        
        # Define output paths
        train_output_path = os.path.join(output_dir, 'train_dataset_info.csv')
        test_output_path = os.path.join(output_dir, 'test_dataset_info.csv')
        
        # Save the splits
        train_df.to_csv(train_output_path, index=False)
        test_df.to_csv(test_output_path, index=False)
        
        print(f"Successfully split dataset:")
        print(f"Total samples: {len(df)}")
        print(f"Training samples: {len(train_df)}")
        print(f"Testing samples: {len(test_df)}")
        print(f"Train file saved to: {train_output_path}")
        print(f"Test file saved to: {test_output_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split dataset_info.csv into train and test sets.')
    parser.add_argument('--input_csv', type=str, help='Path to the input dataset_info.csv file')
    parser.add_argument('--ratio', type=float, default=0.75, help='Ratio of training set (default: 0.75)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    split_dataset(args.input_csv, args.ratio, args.seed)
