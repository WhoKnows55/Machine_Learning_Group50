import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Load the trained model
@st.cache_resource
def load_model():
    with open('XGB_Optuna.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def initialize_default_values():
    # Return a dictionary of all columns with default values
    defaults = {
        'Age at Injury': 30,
        'Alternative Dispute Resolution': 0,
        'Attorney/Representative': 0,
        'Average Weekly Wage': 800.0,
        'COVID-19 Indicator': 0,
        'Gender': 0,  # Male = 1, Female = 0
        'IME-4 Count': 0,
        'Number of Dependents': 0,
        'Missing IME-4 Count': 0,
        'Missing Accident': 0,
        'Missing C-2': 0,
        'Missing C-3': 0,
        'Missing First Hearing': 0,
        'Greenflag Dates': 0,
        'C-2 To Assembly Days': 0,
        'Accident To C-2 Days': 0,
        'Accident To Assembly Days': 0,
        'Follow_The_C2_Rules': 0,
        'Follow_The_C3_Rules': 0,
        'RedFlag_Assembly_to_C-3': 0,
        'RedFlag_Accident_to_C-3': 0,
        'Red_Flag_Accident_To_C-2': 0,
        'Red_Flag_C-2_To_Assembly': 0,
        'Year': datetime.now().year,
        'Season': 0,  # Spring=0, Summer=1, Fall=2, Winter=3
        'Is Weekend': 0,
        'Red_Flag_Year': 0,
        'Red_Flag_Age at Injury': 0,
        'Carrier Name Frequency': 1,
        'District Name Frequency': 1,
        'County of Injury Frequency': 1,
        'Missing Zip Code': 0,
        'Region_I': 0,
        'Region_II': 0,
        'Region_III': 0,
        'Region_IV': 0,
        'Industry Code Avg_Salary': 50000.0,
        'Wage Indicator': 0,
        'is_employeed': 1,
        'Red_Flag_Average Weekly Wage': 0,
        'Missing_Injury': 0,
        'Missing_Body': 0,
        'Severity_Score_Cause': 5.0,
        'Severity_Score_Nature': 5.0,
        'Severity_Score_Part': 5.0
    }
    
    # Add carrier type columns
    for carrier in ['Carrier_1A. PRIVATE', 'Carrier_2A. SIF', 'Carrier_3A. SELF PUBLIC',
                   'Carrier_4A. SELF PRIVATE', 'Carrier_UNKNOWN', 'Carrier_Special_Fund']:
        defaults[carrier] = 0
        
    # Add injury category columns
    for category in ['Creative & Low Risk', 'Customer-Facing Services',
                    'Miscellaneous/Other Services', 'Office-Based Professions',
                    'Physical Labor & High Risk', 'Public Interaction & Environmental Risk',
                    'Unknown']:
        defaults[f'Injury_Category_{category}'] = 0
        
    # Add cause/nature/part columns
    for prefix in ['Cause', 'Nature', 'Part']:
        for i in range(1, 9):
            category_map = {
                1: 'CANCELLED',
                2: 'NON-COMP',
                3: 'MED ONLY',
                4: 'TEMPORARY',
                5: 'PPD SCH LOSS',
                6: 'PPD NSL',
                7: 'PTD',
                8: 'DEATH'
            }
            defaults[f'{i}. {category_map[i]}_{prefix}'] = 0
            
    # Add PTD related columns
    for prefix in ['Cause', 'Nature', 'Part', 'Industry']:
        defaults[f'Has PTD_{prefix}'] = 0
        defaults[f'{prefix}_ptd_pc'] = 0
        
    return defaults

def process_dataframe(df):
    # Make copy to avoid modifying original
    df = df.copy()
    
    # Handle Claim Identifier - convert to numeric hash
    if 'Claim Identifier' in df.columns:
        df['Claim Identifier'] = df['Claim Identifier'].apply(lambda x: hash(str(x)) % 1e8)
    
    # Convert boolean and categorical columns to int
    bool_columns = [col for col in df.columns if df[col].dtype == bool]
    df[bool_columns] = df[bool_columns].astype(int)
    
    # Ensure all other columns are float
    for col in df.columns:
        if df[col].dtype not in ['int32', 'int64', 'float32', 'float64']:
            df[col] = df[col].astype(float)
    
    return df

def main():
    st.title('Workers Compensation Claim Injury Type Prediction')
    st.write('Enter claim details below to predict the injury type.')

    # Get default values
    default_values = initialize_default_values()
    
    # Basic Information Section
    with st.expander("Basic Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input('Age at Injury', min_value=16, max_value=100, value=default_values['Age at Injury'])
            gender = st.selectbox('Gender', ['Female', 'Male'])
            avg_weekly_wage = st.number_input('Average Weekly Wage', min_value=0.0, value=default_values['Average Weekly Wage'])
        with col2:
            num_dependents = st.number_input('Number of Dependents', min_value=0, value=default_values['Number of Dependents'])
            is_employed = st.checkbox('Currently Employed', value=default_values['is_employeed'])
            wage_indicator = st.checkbox('Wage Indicator', value=default_values['Wage Indicator'])

    # Claim Details Section
    with st.expander("Claim Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            adr = st.checkbox('Alternative Dispute Resolution')
            attorney = st.checkbox('Has Attorney/Representative')
            covid_indicator = st.checkbox('COVID-19 Related')
        with col2:
            ime4_count = st.number_input('IME-4 Count', min_value=0, value=default_values['IME-4 Count'])
            missing_ime4 = st.checkbox('Missing IME-4')
    
    # Region and Industry Section
    with st.expander("Region and Industry Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            region = st.selectbox('Region', ['Region I', 'Region II', 'Region III', 'Region IV'])
            industry_category = st.selectbox('Industry Category', [
                'Creative & Low Risk',
                'Customer-Facing Services',
                'Miscellaneous/Other Services',
                'Office-Based Professions',
                'Physical Labor & High Risk',
                'Public Interaction & Environmental Risk',
                'Unknown'
            ])
        with col2:
            carrier_type = st.selectbox('Carrier Type', [
                'PRIVATE',
                'SIF',
                'SELF PUBLIC',
                'SELF PRIVATE',
                'UNKNOWN',
                'Special Fund'
            ])
            industry_salary = st.number_input('Industry Code Average Salary', 
                                            min_value=0.0, 
                                            value=default_values['Industry Code Avg_Salary'])

    # Severity Information
    with st.expander("Severity Scores", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            severity_cause = st.slider('Severity Score (Cause)', 0.0, 10.0, 
                                     value=default_values['Severity_Score_Cause'])
        with col2:
            severity_nature = st.slider('Severity Score (Nature)', 0.0, 10.0, 
                                      value=default_values['Severity_Score_Nature'])
        with col3:
            severity_part = st.slider('Severity Score (Part)', 0.0, 10.0, 
                                    value=default_values['Severity_Score_Part'])

    if st.button('Predict Injury Type'):
        # Start with default values
        input_data = default_values.copy()
        
        # Update with user inputs
        # Generate a unique claim identifier
        claim_id = f'CL{datetime.now().strftime("%Y%m%d%H%M%S")}'
        
        input_data.update({
            'Claim Identifier': claim_id,
            'Age at Injury': age,
            'Gender': 1 if gender == 'Male' else 0,
            'Average Weekly Wage': avg_weekly_wage,
            'Number of Dependents': num_dependents,
            'is_employeed': int(is_employed),
            'Wage Indicator': int(wage_indicator),
            'Alternative Dispute Resolution': int(adr),
            'Attorney/Representative': int(attorney),
            'COVID-19 Indicator': int(covid_indicator),
            'IME-4 Count': ime4_count,
            'Missing IME-4 Count': int(missing_ime4),
            'Industry Code Avg_Salary': industry_salary,
            'Severity_Score_Cause': severity_cause,
            'Severity_Score_Nature': severity_nature,
            'Severity_Score_Part': severity_part
        })
        
        # Update region one-hot encoding
        for r in ['Region_I', 'Region_II', 'Region_III', 'Region_IV']:
            input_data[r] = 1 if f'Region {r[-1]}' == region else 0
            
        # Update industry category one-hot encoding
        for category in ['Creative & Low Risk', 'Customer-Facing Services',
                        'Miscellaneous/Other Services', 'Office-Based Professions',
                        'Physical Labor & High Risk', 'Public Interaction & Environmental Risk',
                        'Unknown']:
            input_data[f'Injury_Category_{category}'] = 1 if category == industry_category else 0
            
        # Update carrier type one-hot encoding
        carrier_mapping = {
            'PRIVATE': 'Carrier_1A. PRIVATE',
            'SIF': 'Carrier_2A. SIF',
            'SELF PUBLIC': 'Carrier_3A. SELF PUBLIC',
            'SELF PRIVATE': 'Carrier_4A. SELF PRIVATE',
            'UNKNOWN': 'Carrier_UNKNOWN',
            'Special Fund': 'Carrier_Special_Fund'
        }
        for display_name, column_name in carrier_mapping.items():
            input_data[column_name] = 1 if carrier_type == display_name else 0

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Define the exact column order from test dataset
        expected_columns = [
            'Claim Identifier', 'Age at Injury', 'Alternative Dispute Resolution',
            'Attorney/Representative', 'Average Weekly Wage', 'COVID-19 Indicator',
            'Gender', 'IME-4 Count', 'Number of Dependents', 'Missing IME-4 Count',
            'Missing Accident', 'Missing C-2', 'Missing C-3', 'Missing First Hearing',
            'Greenflag Dates', 'C-2 To Assembly Days', 'Accident To C-2 Days',
            'Accident To Assembly Days', 'Follow_The_C2_Rules', 'Follow_The_C3_Rules',
            'RedFlag_Assembly_to_C-3', 'RedFlag_Accident_to_C-3',
            'Red_Flag_Accident_To_C-2', 'Red_Flag_C-2_To_Assembly', 'Year',
            'Season', 'Is Weekend', 'Red_Flag_Year', 'Red_Flag_Age at Injury',
            'Carrier Name Frequency', 'District Name Frequency',
            'County of Injury Frequency', 'Missing Zip Code', 'Carrier_1A. PRIVATE',
            'Carrier_2A. SIF', 'Carrier_3A. SELF PUBLIC', 'Carrier_4A. SELF PRIVATE',
            'Carrier_UNKNOWN', 'Carrier_Special_Fund', 'Region_I', 'Region_II',
            'Region_III', 'Region_IV', 'Industry Code Avg_Salary',
            'Injury_Category_Creative & Low Risk',
            'Injury_Category_Customer-Facing Services',
            'Injury_Category_Miscellaneous/Other Services',
            'Injury_Category_Office-Based Professions',
            'Injury_Category_Physical Labor & High Risk',
            'Injury_Category_Public Interaction & Environmental Risk',
            'Injury_Category_Unknown', 'Wage Indicator', 'is_employeed',
            'Red_Flag_Average Weekly Wage', 'Missing_Injury', 'Missing_Body',
            'Severity_Score_Cause', 'Severity_Score_Nature', 'Severity_Score_Part',
            '1. CANCELLED_Cause', '2. NON-COMP_Cause', '3. MED ONLY_Cause',
            '4. TEMPORARY_Cause', '5. PPD SCH LOSS_Cause', '6. PPD NSL_Cause',
            '7. PTD_Cause', '8. DEATH_Cause', '1. CANCELLED_Nature',
            '2. NON-COMP_Nature', '3. MED ONLY_Nature', '4. TEMPORARY_Nature',
            '5. PPD SCH LOSS_Nature', '6. PPD NSL_Nature', '7. PTD_Nature',
            '8. DEATH_Nature', '1. CANCELLED_Part', '2. NON-COMP_Part',
            '3. MED ONLY_Part', '4. TEMPORARY_Part', '5. PPD SCH LOSS_Part',
            '6. PPD NSL_Part', '7. PTD_Part', '8. DEATH_Part', 'Has PTD_Cause',
            'Cause_ptd_pc', 'Has PTD_Nature', 'Nature_ptd_pc', 'Has PTD_Part',
            'Part_ptd_pc', 'Has PTD_Industry', 'Industry_ptd_pc'
        ]
        
        # Ensure DataFrame has all columns in correct order
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[expected_columns]
        
        # Process DataFrame to ensure correct data types
        input_df = process_dataframe(input_df)
        
        try:
            # Load and use the model
            model = load_model()
            prediction = model.predict(input_df)
            
            # Display prediction
            st.success(f'Predicted Injury Type: {prediction[0]}')
            
            # Display feature importance if available
            if hasattr(model, 'feature_importances_'):
                st.subheader('Top 10 Most Important Features')
                importance_df = pd.DataFrame({
                    'Feature': input_df.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                st.bar_chart(importance_df.set_index('Feature'))
                
        except Exception as e:
            st.error(f'Prediction Error: {str(e)}')
            st.write('DataFrame Info:')
            st.write(input_df.dtypes)
            st.write('DataFrame Columns:', input_df.columns.tolist())

if __name__ == '__main__':
    main()