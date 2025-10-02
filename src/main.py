from inspect_data import inspect_dataset
from preprocess import preprocess_dataset, preprocess_text


if __name__ == "__main__":
    print("First step: Inspecting dataset...")
    df = inspect_dataset()

    print("\n Second step: Preprocessing dataset..")
    df = preprocess_dataset(df)
    