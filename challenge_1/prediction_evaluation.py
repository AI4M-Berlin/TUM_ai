import pandas as pd
import numpy as np
from typing import List, Any, Dict
import os
import json

PREDICTION_FILE = 'predictions.csv'
DAYS_AFTER_SYMPTOM_TO_OMIT = 5
DAYS_BEFORE_SYMPTOMS = 5
LABEL_FILE = 'labels.csv'
TEST_DATA_FOLDER = 'test'

def get_real_labels(label_file: str) -> pd.DataFrame:
  """Fetch real test labels test_set.csv as dataframe 

     Returns the test labels as a DataFrame
  """
  labels = pd.read_csv(label_file, header=0)
  return labels

def get_test_data(case: str) -> pd.DataFrame:
  """Fetch real test data as dataframe for each case

     Args:
        case: test case which points to the name of the file

     Returns the test data as a DataFrame
  """
  data_file_path = os.path.join(TEST_DATA_FOLDER, f"{case}.csv")
  test_data = pd.read_csv(data_file_path, header=0)
  return test_data

def get_num_days(data: pd.DataFrame) -> int:
  """Given the heartrate data with timestamp return the number of days
     
     Args:
        data: test data of a particular case

     Returns the number of days
  """
  first_date = pd.to_datetime(data['timestamp'].iloc[0]).date()
  last_date = pd.to_datetime(data['timestamp'].iloc[-1]).date()
  return (last_date - first_date).days 


def evaluate_fp_case(preds: pd.Series, true: str, case: str) -> float:
  """Calculate the False Positive Rate for the given case with the formula:
     false_positives / (false_positives + true_negatives)

     Args:
        preds: all the predictions made for symptom onset date
        true: true symptom onset date
        case: the test case that is to be evaluated

     Returns the False positive rate for the test case
  """
  fp_count = 0
  tn_count = 0

  days = get_num_days(get_test_data(case))
  gt_date = pd.to_datetime(true).date()
  try:
    for i, p in enumerate(preds):
      p_date = pd.to_datetime(p).date()
      if ((gt_date - p_date).days > DAYS_BEFORE_SYMPTOMS) or ((p_date - gt_date).days > DAYS_AFTER_SYMPTOM_TO_OMIT):
        fp_count += 1
  except:
    print('No prediction available for this case')
    return 0
  
  tn_count = days - fp_count - 11

  try:
    return fp_count/(fp_count+tn_count)
  except ZeroDivisionError:
    return 0


def evaluate_tp_case(preds: pd.Series, true: str) -> int:
  """Calculate the true positive status for the given case which is when at least one day between
     DAYS_BEFORE_SYMPTOM to symptom_onset date is predicted. 

     Args:
        preds: all the predictions made for symptom onset date
        true: true symptom onset date

     Returns the true positive status for the test case
  """
  count = 0
  gt_date = pd.to_datetime(true).date()
  
  try:
    for i, p in enumerate(preds):
      p_date = pd.to_datetime(p).date()
      if ((gt_date - p_date).days <= DAYS_BEFORE_SYMPTOMS) and ((gt_date - p_date).days >= 0):
        count += 1
  except:
    print('No prediction available for this case')
  
  return 1 if count >= 1 else 0


def save_results(result: Dict[str, Any]) -> None:
  """Final results are written to the directory 'results'
    
     Args:
        result: contains both cummulative and per case results as a dictionary
  """
  result_json = json.dumps(result)
  with open(os.path.join('results', 'metrics.json'), 'w') as outfile:
    json.dump(result_json, outfile)

  
def evaluate_prediction(prediction_file, label_file):
  """Evaluates the predictions and saves the evaluations in the folder 'results/result.json'
  Args:
        event (dict): Event payload.
        context (google.cloud.functions.Context): Metadata for the event.
  """
  
  preds_df = pd.read_csv(prediction_file, header=0)
  ground_truth = get_real_labels(label_file)
  tp_per_case = []
  tp_per_case_dict = {}
  fp_per_case = []
  fp_per_case_dict = {}
  for i in range(ground_truth.shape[0]):
    case = ground_truth['case'][i]

    preds_case = preds_df[preds_df['case']==case]['symptom_onset']
    ground_truth_case = ground_truth[ground_truth['case']==case]['symptom_onset'].iloc[0]

    tp_metric = evaluate_tp_case(preds_case, ground_truth_case)
    tp_per_case.append(tp_metric)
    tp_per_case_dict[case] = tp_metric

    fp_metric = evaluate_fp_case(preds_case, ground_truth_case, case)
    fp_per_case.append(fp_metric)
    fp_per_case_dict[case] = fp_metric

  result = {}
  result['true_positive_rate'] = np.sum(np.array(tp_per_case))/ground_truth.shape[0]
  result['false_positive_rate'] = np.mean(np.array(fp_per_case))
  result['combined_btpr_fpr'] = (result['true_positive_rate'] + (1-result['false_positive_rate']))/2
  result['true_positive_per_case'] = tp_per_case_dict
  result['false_positive_per_case'] = fp_per_case_dict

  save_results(result)

if __name__ == '__main__':
    evaluate_prediction(PREDICTION_FILE, LABEL_FILE)