import pandas as pd
import numpy as np

def generate_investigate_first_queue(df, y_pred, threshold_minutes=45.0):
    df_eval = df.copy()
    df_eval['predicted_wait'] = y_pred
    
    dept_baselines = df_eval.groupby('department_id')['wait_minutes'].transform('median')
    df_eval['baseline_wait'] = dept_baselines
    df_eval['excess_wait'] = np.maximum(0, df_eval['predicted_wait'] - df_eval['baseline_wait'])
    df_eval['hour_slot'] = df_eval['arrival_time'].dt.hour
    
    summary = df_eval.groupby(['department_id', 'department_name', 'hour_slot']).agg(
        affected_encounters=('encounter_id', 'count'),
        avg_predicted_wait=('predicted_wait', 'mean'),
        avg_baseline_wait=('baseline_wait', 'mean'),
        avg_excess_wait=('excess_wait', 'mean')
    ).reset_index()
    
    summary['impact_score'] = summary['avg_excess_wait'] * summary['affected_encounters']
    summary = summary.sort_values('impact_score', ascending=False).reset_index(drop=True)
    
    def get_action_check(row):
        if row['avg_predicted_wait'] > threshold_minutes and row['affected_encounters'] > 10:
            return "High surge volume: Review staffing shifts & room capacity"
        elif row['avg_excess_wait'] > 15:
            return "Severe bottleneck: Investigate diagnostic test turnaround time"
        else:
            return "Standard check: Review appointment scheduling"
            
    summary['suggested_action'] = summary.apply(get_action_check, axis=1)
    return summary
