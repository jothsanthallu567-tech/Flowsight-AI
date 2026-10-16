import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Palette Constants
TEAL = "#0284C7"
TEAL_LIGHT = "#E0F2FE"
DARK_NAVY = "#0F172A"
MUTED_SLATE = "#64748B"
BORDER_COLOR = "#E2E8F0"
GRID_COLOR = "#F1F5F9"
ALERT_RED = "#EF4444"
ALERT_AMBER = "#F59E0B"
ALERT_GREEN = "#10B981"

def apply_clean_theme(fig, title=""):
    """Applies a minimalist, modern, high-contrast healthcare tech theme to Plotly figures."""
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=14, color=DARK_NAVY, family="Inter, sans-serif"),
            x=0.02,
            y=0.96
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=35, r=20, t=45 if title else 20, b=35),
        font=dict(family="Inter, -apple-system, sans-serif", color=MUTED_SLATE, size=11),
        hoverlabel=dict(
            bgcolor=DARK_NAVY,
            font_size=12,
            font_family="Inter, sans-serif",
            font_color="#FFFFFF"
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=False,
            color=MUTED_SLATE,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=False,
            color=MUTED_SLATE,
            tickfont=dict(size=10)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        )
    )
    return fig

def plot_arrival_volume_clean(df):
    """Sleek 2025 daily volume trend with area fill."""
    df_ts = df.copy()
    df_ts['date'] = df_ts['arrival_time'].dt.date
    daily = df_ts.groupby('date')['encounter_id'].count().reset_index()
    daily.columns = ['Date', 'Encounters']
    
    # 7-day rolling average for smooth trendline
    daily['7D_MA'] = daily['Encounters'].rolling(7, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily['Date'],
        y=daily['Encounters'],
        mode='lines',
        name='Daily Volume',
        line=dict(color="#38BDF8", width=1.5),
        opacity=0.6
    ))
    fig.add_trace(go.Scatter(
        x=daily['Date'],
        y=daily['7D_MA'],
        mode='lines',
        name='7-Day Trend',
        line=dict(color=TEAL, width=3),
        fill='tozeroy',
        fillcolor='rgba(2, 132, 199, 0.08)'
    ))
    fig = apply_clean_theme(fig, "Patient Arrival Trends (Full Year 2025)")
    fig.update_yaxes(title_text="Visits / Day")
    return fig

def plot_dept_wait_bars_clean(df):
    """Clean horizontal bar chart of average wait time by department."""
    dept_summary = df.groupby('department_name')['wait_minutes'].mean().reset_index()
    dept_summary.columns = ['Department', 'AvgWait']
    dept_summary = dept_summary.sort_values('AvgWait', ascending=True)

    fig = px.bar(
        dept_summary,
        x='AvgWait',
        y='Department',
        orientation='h',
        color='AvgWait',
        color_continuous_scale=[[0, '#38BDF8'], [0.5, '#F59E0B'], [1, '#EF4444']],
        labels={'AvgWait': 'Avg Wait (min)', 'Department': ''}
    )
    fig.update_traces(marker_line_width=0, opacity=0.9)
    fig = apply_clean_theme(fig, "Average Wait Time by Department (Minutes)")
    fig.update_coloraxes(showscale=False)
    return fig

def plot_hourly_heatmap_clean(df):
    """Clear heatmap of arrival volume across Hour of Day vs Day of Week."""
    df_hm = df.copy()
    df_hm['hour'] = df_hm['arrival_time'].dt.hour
    df_hm['day'] = df_hm['arrival_time'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    pivot = df_hm.pivot_table(index='day', columns='hour', values='encounter_id', aggfunc='count', fill_value=0)
    pivot = pivot.reindex(index=day_order, columns=list(range(24)), fill_value=0)
    
    fig = px.imshow(
        pivot,
        labels=dict(x="Hour of Day", y="", color="Visits"),
        x=[f"{h:02d}:00" for h in range(24)],
        y=day_order,
        color_continuous_scale="Blues",
        aspect="auto"
    )
    fig = apply_clean_theme(fig, "Hospital Arrival Density (Hour of Day vs. Day of Week)")
    return fig

def plot_triage_box_clean(df):
    """Clean box plot of wait time by triage priority."""
    fig = px.box(
        df,
        x='triage_level',
        y='wait_minutes',
        color='triage_level',
        color_discrete_map={'Emergency': ALERT_RED, 'High': '#F97316', 'Medium': TEAL, 'Low': ALERT_GREEN},
        category_orders={'triage_level': ['Emergency', 'High', 'Medium', 'Low']},
        labels={'triage_level': 'Triage Priority', 'wait_minutes': 'Wait Time (min)'}
    )
    fig.update_layout(showlegend=False)
    fig = apply_clean_theme(fig, "Wait Time Distribution by Triage Priority")
    return fig

def plot_model_calibration_clean(actual, predicted):
    """Scatter calibration plot with ideal 1:1 line."""
    eval_df = pd.DataFrame({'Actual': actual, 'Predicted': predicted})
    sample = eval_df.sample(min(400, len(eval_df)), random_state=42)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample['Actual'],
        y=sample['Predicted'],
        mode='markers',
        marker=dict(color=TEAL, size=6, opacity=0.55),
        name='Encounters'
    ))
    max_val = max(eval_df['Actual'].max(), eval_df['Predicted'].max())
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(color=ALERT_RED, dash='dash', width=2),
        name='Ideal 1:1 Line'
    ))
    fig = apply_clean_theme(fig, "Model Calibration: Predicted vs. Actual Wait (Test Set)")
    fig.update_xaxes(title_text="Actual Wait Time (min)")
    fig.update_yaxes(title_text="Predicted Wait Time (min)")
    return fig

def plot_loss_curve_clean(loss_history):
    """Training loss curve showing Gradient Descent convergence."""
    step = max(1, len(loss_history) // 150)
    indices = list(range(0, len(loss_history), step))
    sampled_loss = [loss_history[i] for i in indices]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=indices,
        y=sampled_loss,
        mode='lines',
        line=dict(color=TEAL, width=2.5),
        name='MSE Loss'
    ))
    fig = apply_clean_theme(fig, "NumPy Gradient Descent Convergence (Loss vs. Iterations)")
    fig.update_xaxes(title_text="Iteration (0 to 1,500)")
    fig.update_yaxes(title_text="Mean Squared Error (MSE)")
    return fig

def plot_tat_clean(diagnostic_tests):
    """Diagnostic Test TAT breakdown."""
    tat_agg = diagnostic_tests.groupby(['test_type', 'priority'])['tat_minutes'].mean().reset_index()
    fig = px.bar(
        tat_agg,
        x='test_type',
        y='tat_minutes',
        color='priority',
        barmode='group',
        color_discrete_map={'Urgent': ALERT_RED, 'High': ALERT_AMBER, 'Routine': TEAL},
        category_orders={'priority': ['Urgent', 'High', 'Routine']},
        labels={'test_type': 'Test Type', 'tat_minutes': 'Avg Turnaround (min)', 'priority': 'Priority'}
    )
    fig = apply_clean_theme(fig, "Diagnostic Test Turnaround Time by Modality")
    return fig
