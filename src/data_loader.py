import pandas as pd
import numpy as np

def load_raw_data(data_dir="data"):
    patients = pd.read_csv(f"{data_dir}/patients.csv")
    encounters = pd.read_csv(f"{data_dir}/encounters.csv")
    departments = pd.read_csv(f"{data_dir}/departments.csv")
    diagnostic_tests = pd.read_csv(f"{data_dir}/diagnostic_tests.csv")
    return patients, encounters, departments, diagnostic_tests

def build_merged_dataset(data_dir="data"):
    patients, encounters, departments, diagnostic_tests = load_raw_data(data_dir)
    
    # Impute missing patient condition
    patients['chronic_condition'] = patients['chronic_condition'].fillna('None')
    patients['has_chronic_condition'] = (patients['chronic_condition'] != 'None').astype(int)
    
    # Parse timestamps
    encounters['arrival_time'] = pd.to_datetime(encounters['arrival_time'])
    encounters['service_start_time'] = pd.to_datetime(encounters['service_start_time'])
    encounters['service_end_time'] = pd.to_datetime(encounters['service_end_time'])
    encounters['discharge_time'] = pd.to_datetime(encounters['discharge_time'])
    
    encounters = encounters.sort_values('arrival_time').reset_index(drop=True)
    
    # Aggregate diagnostic tests to encounter level to prevent row duplication
    diagnostic_summary = diagnostic_tests.groupby('encounter_id').agg(
        num_diagnostic_tests=('test_id', 'count'),
        has_high_priority_test=('priority', lambda x: int(('High' in x.values) or ('Urgent' in x.values)))
    ).reset_index()
    
    # Join tables cleanly
    df = encounters.merge(patients, on='patient_id', how='left')
    df = df.merge(departments, on='department_id', how='left')
    df = df.merge(diagnostic_summary, on='encounter_id', how='left')
    
    df['num_diagnostic_tests'] = df['num_diagnostic_tests'].fillna(0).astype(int)
    df['has_high_priority_test'] = df['has_high_priority_test'].fillna(0).astype(int)
    
    return df
