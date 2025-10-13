import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(page_title="Traffic Accidents Dashboard", layout="wide")


# Load the data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("traffic_accidents.csv.gz", compression="gzip")
 
        column_mapping = {
            'crash_date': 'crash_date',
            'crash_hour': 'hour',
            'crash_day_of_week': 'day_of_week',
            'crash_month': 'month',
            'injuries_total': 'injuries_total',
            'injuries_fatal': 'fatalities_total',
            'roadway_surface_cond': 'roadway_surface_cond',
            'lighting_condition': 'lighting_condition',
            'crash_type': 'crash_type',
            'num_units': 'num_units',
            'prim_contributory_cause': 'primary_contributory_cause',
            'intersection_related_i': 'intersection_related',
            'weather_condition': 'weather'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Data cleaning
        df['crash_date'] = pd.to_datetime(df['crash_date'], errors='coerce')
        df['year'] = df['crash_date'].dt.year
        df['month'] = df['month'].fillna(1).astype(int)
        df['hour'] = df['hour'].fillna(0).astype(int)
        df['injuries_total'] = pd.to_numeric(df['injuries_total'], errors='coerce').fillna(0)
        df['num_units'] = pd.to_numeric(df['num_units'], errors='coerce').fillna(1)
        

        df['intersection_related'] = df['intersection_related'].fillna('N')
        df['is_intersection'] = df['intersection_related'].str.upper().str[0] == 'Y'
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Main app
def main():
    st.title("🚗 Traffic Accidents Dashboard")
    
    # Load data
    df = load_data()

    if df.empty:
        st.error("No data loaded. Please check your CSV file.")
        return


    st.sidebar.header("Filters")


    available_years = sorted(df['year'].unique())
    selected_years = st.sidebar.multiselect(
        "Year", 
        available_years, 
        default=available_years
    )

    months = ['All'] + list(range(1, 13))
    selected_month = st.sidebar.selectbox("Month", months)


    days = ['All'] + ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    selected_day = st.sidebar.selectbox("Day of Week", days)

    weather_conditions = ['All'] + sorted(df['weather'].dropna().unique().tolist())
    selected_weather = st.sidebar.selectbox("Weather Condition", weather_conditions)


    lighting_conditions = ['All'] + sorted(df['lighting_condition'].dropna().unique().tolist())
    selected_lighting = st.sidebar.selectbox("Lighting Condition", lighting_conditions)


    surface_conditions = ['All'] + sorted(df['roadway_surface_cond'].dropna().unique().tolist())
    selected_surface = st.sidebar.selectbox("Roadway Surface Condition", surface_conditions)


    intersection_options = ['All', 'Yes', 'No']
    selected_intersection = st.sidebar.selectbox("Intersection Related", intersection_options)

    filtered_df = df.copy()

    if selected_years:
        filtered_df = filtered_df[filtered_df['year'].isin(selected_years)]

    if selected_month != 'All':
        filtered_df = filtered_df[filtered_df['month'] == selected_month]

    if selected_day != 'All':
        day_mapping = {day: idx for idx, day in enumerate(days[1:], 1)}
        filtered_df = filtered_df[filtered_df['day_of_week'] == day_mapping[selected_day]]

    if selected_weather != 'All':
        filtered_df = filtered_df[filtered_df['weather'] == selected_weather]

    if selected_lighting != 'All':
        filtered_df = filtered_df[filtered_df['lighting_condition'] == selected_lighting]

    if selected_surface != 'All':
        filtered_df = filtered_df[filtered_df['roadway_surface_cond'] == selected_surface]

    if selected_intersection != 'All':
        is_intersection = selected_intersection == 'Yes'
        filtered_df = filtered_df[filtered_df['is_intersection'] == is_intersection]

    # PART A - Overview & Descriptives
    st.header("Overview & Descriptives")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    total_crashes = len(filtered_df)
    total_injuries = filtered_df['injuries_total'].sum()
    avg_injuries_per_crash = total_injuries / total_crashes if total_crashes > 0 else 0
    intersection_pct = (filtered_df['is_intersection'].sum() / total_crashes * 100) if total_crashes > 0 else 0

    with col1:
        st.metric("Total Crashes", f"{total_crashes:,}")

    with col2:
        st.metric("Total Injuries", f"{total_injuries:,}")

    with col3:
        st.metric("Avg Injuries/Crash", f"{avg_injuries_per_crash:.2f}")

    with col4:
        st.metric("Intersection Related", f"{intersection_pct:.1f}%")

    # Summary table - Crashes by day_of_week and month
    st.subheader("Crashes by Day of Week and Month")

    col1, col2 = st.columns(2)

    with col1:

        day_summary = filtered_df['day_of_week'].value_counts().sort_index()
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_df = pd.DataFrame({
            'Day': [day_names[i-1] if i <= len(day_names) else f'Day {i}' for i in day_summary.index],
            'Crashes': day_summary.values
        }).sort_values('Crashes', ascending=False)
        
        st.dataframe(day_df.head(3).style.highlight_max(axis=0, color='lightgreen'), 
                    use_container_width=True)

    with col2:

        month_summary = filtered_df['month'].value_counts().sort_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_df = pd.DataFrame({
            'Month': [month_names[i-1] if i <= len(month_names) else f'Month {i}' for i in month_summary.index],
            'Crashes': month_summary.values
        }).sort_values('Crashes', ascending=False)
        
        st.dataframe(month_df.head(3).style.highlight_max(axis=0, color='lightgreen'), 
                    use_container_width=True)

    # PART B - When do crashes happen?
    st.header("When Do Crashes Happen?")

    col1, col2 = st.columns(2)

    with col1:

        day_crashes = filtered_df['day_of_week'].value_counts().sort_index()
        day_crashes.index = [day_names[i-1] if i <= len(day_names) else f'Day {i}' for i in day_crashes.index]
        day_crashes_sorted = day_crashes.sort_values(ascending=False)
        
        max_day = day_crashes_sorted.index[0]
        max_crashes = day_crashes_sorted.iloc[0]
        
        fig_day = px.bar(x=day_crashes_sorted.index, y=day_crashes_sorted.values,
                        title=f'Crashes by Day of Week (Highest: {max_day})',
                        labels={'x': 'Day of Week', 'y': 'Number of Crashes'})
        fig_day.update_traces(text=day_crashes_sorted.values, textposition='auto')
        st.plotly_chart(fig_day, use_container_width=True)

    with col2:

        heatmap_data = filtered_df.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        heatmap_data.index = [day_names[i-1] if i <= len(day_names) else f'Day {i}' for i in heatmap_data.index]
        
        fig_heatmap = px.imshow(heatmap_data, 
                               title='Crash Heatmap: Hour × Day of Week',
                               labels=dict(x="Hour", y="Day of Week", color="Crashes"),
                               aspect="auto")
        

        max_val = heatmap_data.max().max()
        max_coords = np.where(heatmap_data.values == max_val)
        peak_day = heatmap_data.index[max_coords[0][0]]
        peak_hour = heatmap_data.columns[max_coords[1][0]]
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.caption(f"Peak: {peak_day} at {peak_hour:02d}:00 ({max_val} crashes)")


    monthly_trend = filtered_df.groupby('month').size()
    monthly_trend.index = [month_names[i-1] if i <= len(month_names) else f'Month {i}' for i in monthly_trend.index]

    fig_month = px.line(x=monthly_trend.index, y=monthly_trend.values,
                       title='Crashes by Month',
                       labels={'x': 'Month', 'y': 'Number of Crashes'})
    fig_month.update_traces(mode='lines+markers')

    max_month = monthly_trend.idxmax()
    min_month = monthly_trend.idxmin()
    fig_month.add_annotation(x=max_month, y=monthly_trend[max_month],
                            text=f"Peak: {max_month}",
                            showarrow=True, arrowhead=1)
    fig_month.add_annotation(x=min_month, y=monthly_trend[min_month],
                            text=f"Low: {min_month}",
                            showarrow=True, arrowhead=1)

    st.plotly_chart(fig_month, use_container_width=True)


    st.header("Severity & Conditions")

    col1, col2 = st.columns(2)

    with col1:

        surface_injuries = filtered_df[filtered_df['injuries_total'] > 0]
        
        if not surface_injuries.empty:
            fig_box = px.box(surface_injuries, 
                            x='roadway_surface_cond', 
                            y='injuries_total',
                            title='Injury Severity by Roadway Surface Condition',
                            labels={'roadway_surface_cond': 'Surface Condition', 
                                   'injuries_total': 'Total Injuries'})
            

            medians = surface_injuries.groupby('roadway_surface_cond')['injuries_total'].median()
            highest_median_condition = medians.idxmax() if not medians.empty else "N/A"
            
            st.plotly_chart(fig_box, use_container_width=True)
            st.caption(f"Highest median injuries: {highest_median_condition}")
        else:
            st.info("No injury data available for selected filters")

    with col2:
        if not filtered_df.empty:
            crash_lighting = pd.crosstab(filtered_df['lighting_condition'], 
                                       filtered_df['crash_type'], 
                                       normalize='index') * 100
            
            fig_stacked = px.bar(crash_lighting, 
                                title='Crash Type Distribution by Lighting Condition',
                                labels={'value': 'Percentage', 
                                       'lighting_condition': 'Lighting Condition',
                                       'variable': 'Crash Type'},
                                barmode='stack')
            
            st.plotly_chart(fig_stacked, use_container_width=True)
            

            if len(crash_lighting) > 1:
                st.caption("Compare how crash type distributions change between daylight and nighttime conditions")
        else:
            st.info("No data available for selected filters")


    scatter_data = filtered_df[filtered_df['num_units'] <= 10] 

    if not scatter_data.empty:
        fig_scatter = px.scatter(scatter_data, 
                                x='num_units', 
                                y='injuries_total',
                                title='Number of Vehicles vs Total Injuries',
                                labels={'num_units': 'Number of Vehicles', 
                                       'injuries_total': 'Total Injuries'},
                                trendline="lowess")

        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("Multi-vehicle crashes tend to result in more injuries, showing positive correlation")
    else:
        st.info("No data available for scatter plot")

    # PART D - Contributors & Locations
    st.header("📍 Contributors & Locations")

    col1, col2 = st.columns(2)

    with col1:
        # Horizontal bar - top 10 contributory causes
        top_causes = filtered_df['primary_contributory_cause'].value_counts().head(10)
        
        if not top_causes.empty:
            top_cause = top_causes.index[0]
            top_count = top_causes.iloc[0]
            
            fig_causes = px.bar(x=top_causes.values, y=top_causes.index,
                               orientation='h',
                               title=f'Top 10 Contributory Causes (#1: {top_cause})',
                               labels={'x': 'Number of Crashes', 'y': 'Cause'})
            fig_causes.update_traces(text=top_causes.values, textposition='auto')
            st.plotly_chart(fig_causes, use_container_width=True)
        else:
            st.info("No contributory cause data available")

    with col2:

        intersection_counts = filtered_df['is_intersection'].value_counts()
        
        if not intersection_counts.empty:
            intersection_labels = {True: 'Intersection', False: 'Non-Intersection'}
            intersection_counts.index = [intersection_labels.get(x, x) for x in intersection_counts.index]
            
            fig_pie = px.pie(values=intersection_counts.values,
                            names=intersection_counts.index,
                            title='Intersection vs Non-Intersection Crashes',
                            hole=0.4)
            fig_pie.update_traces(textinfo='percent+label')
            
            st.plotly_chart(fig_pie, use_container_width=True)
            
            intersection_pct = (intersection_counts.get('Intersection', 0) / intersection_counts.sum() * 100)
            st.caption(f"{intersection_pct:.1f}% of crashes are intersection-related")
        else:
            st.info("No intersection data available")


    st.header("Key Insights")

    insights = f"""
    **1. Which day of the week has the most crashes, and during which hours do peaks occur?**
    - **Most crashes:** {max_day}
    - **Peak hours:** {peak_day} at {peak_hour:02d}:00 ({max_val} crashes)

    **2. Are crashes seasonal?**
    - **Peak month:** {max_month}
    - **Lowest month:** {min_month}
    - **Seasonality:** {"Crashes show clear seasonality" if abs(monthly_trend[max_month] - monthly_trend[min_month]) > monthly_trend.mean() * 0.3 else "Limited seasonality observed"}

    **3. Under which roadway surface condition is injury severity highest?**
    - **Highest median injuries:** {highest_median_condition if 'highest_median_condition' in locals() else 'N/A'}

    **4. How does lighting condition change the mix of crash types?**
    - Compare the stacked bar chart to see how crash type distributions shift between different lighting conditions

    **5. What share of crashes are intersection-related, and what's the top primary contributory cause?**
    - **Intersection-related:** {intersection_pct:.1f}%
    - **Top contributory cause:** {top_cause if 'top_cause' in locals() else 'N/A'}
    """

    st.write(insights)


if __name__ == "__main__":
    main()