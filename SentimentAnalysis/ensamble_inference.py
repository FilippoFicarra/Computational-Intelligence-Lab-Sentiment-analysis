import csv

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from datasets import TwitterDatasetEnsambleTest
from ensamble_train import LinearCombinationModel
from utility_functions import *

if __name__ == "__main__":
    # Get test dataframe
    df = pd.read_json(PATH_DATASET_TWITTER_TEST, lines=True, dtype=DTYPE_TWITTER_TEST)

    # Create tokeinizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Create dataset and data loader
    dataset = TwitterDatasetEnsambleTest(df, tokenizer)
    loader = DataLoader(dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Upload all ensamble_models.
    ensamble_models = []
    e_model = None
    name_ensamble = None
    # Check if the folder exists
    if not os.path.exists(PATH_MODELS):
        raise Exception("Folder with models does not exist.")
    else:
        # Check if the folder is empty
        files_in_folder = os.listdir(PATH_MODELS)
        if not files_in_folder:
            raise Exception("Folder is empty.")
        else:
            # Filter the files to get only the ones ending with ".pt"
            pt_files = [file for file in files_in_folder if file.endswith(".pt")]

            if len(pt_files) != 4:
                raise Exception("This module only works with ensamble of three models. Either the model with the "
                                "weights for the ensamble or one of the models of the ensamble is missing.")

            else:
                # Loop over the .pt files in the folder
                for pt_file in sorted(pt_files):
                    if "ensamble" in pt_file:
                        e_model = LinearCombinationModel()
                        e_model.load_model(os.path.join(PATH_MODELS, pt_file))
                        e_model.eval()
                        e_model.to(device)
                        name_ensamble = pt_file.replace(".pt", "")
                    else:
                        # Get model
                        trained_model, mask = get_model(pt_file, device)
                        # Set model for evaluation
                        trained_model.eval()
                        ensamble_models.append((trained_model, mask))

    if e_model is None:
        raise Exception("No model for ensamble found...")

    # Define list of predictions
    all_predictions = []

    # Compute predictions
    with torch.no_grad():
        for sample in iter(loader):
            ids = sample['input_ids'].to(device)
            mask = sample['attention_mask'].to(device)
            ids_masker = sample['input_ids_masker'].to(device)
            mask_masker = sample['attention_mask_masker'].to(device)

            # Compute outputs of the models in the ensamble
            outputs = []
            for model, requires_mask in ensamble_models:
                if requires_mask:
                    outputs.append(model(ids_masker, mask_masker))
                else:
                    outputs.append(model(ids, mask))

            # Compute model output
            prediction_ensamble = e_model(outputs[0], outputs[1], outputs[2])

            all_predictions.append(torch.argmax(prediction_ensamble, dim=1))

    # Produce final tensor with predictions
    all_predictions = torch.cat(all_predictions, dim=0)
    all_predictions[all_predictions == 0] = -1

    # Save predictions to file
    data = [(idx + 1, value.item()) for idx, value in enumerate(all_predictions)]

    # Check if the directory for predictions already exists
    if not os.path.exists(PATH_PREDICTIONS):
        # If it doesn't exist, create the directory
        os.makedirs(PATH_PREDICTIONS)

    # Write the data to a CSV file
    with open(PATH_PREDICTIONS + "/" + name_ensamble + ".csv", "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Id', 'Prediction'])
        csv_writer.writerows(data)

    print("- predictions saved!")
