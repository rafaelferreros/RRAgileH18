import pandas as pd
import numpy as np
import config as cfg


def normalize_data(dataset):
    """Normalize the data for some known border values.

    Param: dataset The data to work with.

    Return: The normalized dataset.
    """
    output = dataset[(dataset[cfg.age_header] > 0)
                     & (dataset[cfg.income_header] <= 30000)
                     & (dataset[cfg.days_overdue_header] <= 1000)]

    return output


def load_data():
    """Transfor the input dataset to the model.

    Param:  csv_file The dataset filename.

    Return: data               The dataset with data to use in the model.
    Return: id                 The dataset (column) with the ide of each entry
                               in the model.
    Return: train_target_data: The dataset (column) with the results entries
                               that should be used for training.
    """
    use_columns = [cfg.id_header, cfg.days_overdue_header,
                   cfg.current_job_days_header, cfg.age_header,
                   cfg.income_header, cfg.training_target_header]

    data = pd.read_csv(cfg.input_dataset_filename, usecols=use_columns)
    data = normalize_data(data)

    id_data = data[cfg.id_header]
    train_target_data = data[cfg.training_target_header]
    output_data = data.drop([cfg.id_header, cfg.training_target_header], axis=1)

    return output_data, id_data, train_target_data
