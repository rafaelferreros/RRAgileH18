"""Configuration module

This module describes the configuration variables used:

Variables used for training:

training_dataset_filename: The filename of the csv data to use for training.
training_random_seed:      A numeric value used as random seed to split the
                           training data.
training_test_percent:     The percent of the training data that will be used
                           for validating and messuring the learning process.
training_target_header:    The column with the target that for training the
                           algorithm.

Variables use as input for the algorithm:

input_dataset_filename:  The filename of the input csv data file to process.
output_dataset_filename: The filename of the output csv datat file to write.
id_header:               The column name with the unique identificator of each
                         entry.
days_overdue_header:     The column name with the overdue days.
current_job_days_header: The column name with the days in the current job.
age_header:              The column name with the age.
income_header:           The column name with the monthly income.

"""
training_random_seed  = 100
training_test_percent = 0.60

training_dataset_filename = '../dataset/hackaton_training_v1.csv'
training_target_header    = 'v_10'

input_dataset_filename  = '../dataset/hackaton_training_v1.csv'
id_header               = 'v_0'
days_overdue_header     = 'v_1'
current_job_days_header = 'v_2'
age_header              = 'v_4'
income_header           = 'v_6'
output_dataset_filename = '../result/result.csv'

