import pandas as pd
import config as cfg
import os.path

def save_results(id, data):
    """Save the results

    Param: id   The dataset (column) of the id for each entry.
    Param: data The dataset (column) with the result.
    """
    assert(len(id) == len(data))

    id_data = id.to_frame()
    predicted_data = pd.DataFrame(data)

    result_data = id_data.join(predicted_data)

    print(result_data.tail())
    return

    result_path = os.path.dirname(cfg.output_dataset_filename);
    try:
        os.mkdir(result_path)
    except FileExistsError:
        pass

    result_data.to_csv(cfg.output_dataset_filename)
