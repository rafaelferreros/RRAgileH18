import etl
import algo
import result
import config as cfg

def main():
    dataset, data_id, train_target = etl.load_data()

    model = algo.create_random_forest_regressor()
    algo.train_model(model, dataset, train_target,
                     cfg.training_test_percent, cfg.training_random_seed)

    predicted_data = algo.predict(model, dataset)

    result.save_results(data_id, predicted_data)

if __name__ == '__main__':
    main()
