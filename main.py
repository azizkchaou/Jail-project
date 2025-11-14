import pandas as pd
import numpy as np

# Read CSV with low_memory=False to reduce dtype warnings for large files
sent = pd.read_csv('Sentencing.csv', low_memory=False)

# Basic inspection
#print(sent.head())
#sent.info()

# Check for missing values (percent)
missing_percent = (sent.isnull().sum() / len(sent)) * 100
#print(missing_percent)

# Filter dataset - keep only rows where BOTH COMMITMENT_TYPE and COMMITMENT_TERM are present
sent_filtered = sent[(sent['COMMITMENT_TYPE'].notna()) & (sent['COMMITMENT_TERM'].notna())].copy()
# print(f"Original dataset shape: {sent.shape}")
# print(f"Filtered dataset (both commitment fields present) shape: {sent_filtered.shape}")

# Normalize commitment unit text early so comparisons are reliable
sent_filtered['COMMITMENT_UNIT'] = (
    sent_filtered['COMMITMENT_UNIT'].fillna('').astype(str).str.strip().str.upper()
)

# print('COMMITMENT_TYPE counts:')
# print(sent_filtered['COMMITMENT_TYPE'].value_counts())
# print('\nCOMMITMENT_UNIT unique examples:')
# print(sent_filtered['COMMITMENT_UNIT'].unique()[:50])

# Drop rows where COMMITMENT_UNIT is in the ignore list
units_to_ignore = ['POUNDS', 'OUNCES', 'KILOS', 'GRAMS', 'DOLLARS', 'TERM']
#print('\nRows with ignored units:', sent_filtered['COMMITMENT_UNIT'].isin(units_to_ignore).sum())
sent_filtered = sent_filtered[~sent_filtered['COMMITMENT_UNIT'].isin(units_to_ignore)].copy()
#print(f"After dropping ignored units: {sent_filtered.shape}")

# Convert COMMITMENT_TERM to numeric (coerce errors to NaN) before computing years
sent_filtered['COMMITMENT_TERM_NUM'] = pd.to_numeric(sent_filtered['COMMITMENT_TERM'], errors='coerce')

# Map units to year factors (make sure keys are uppercase, normalized)
unit_to_years = {
    'DAY': 1/365,
    'DAYS': 1/365,
    'WEEK': 7/365,
    'WEEKS': 7/365,
    'MONTH': 1/12,
    'MONTHS': 1/12,
    'YEAR': 1,
    'YEARS': 1,
    'YEAR(S)': 1,
    'HOUR': 1/(24*365),
    'HOURS': 1/(24*365),
    # special cases
}

sent_filtered['unit_factor'] = sent_filtered['COMMITMENT_UNIT'].map(unit_to_years)

# Compute sentence years normally (term * unit factor)
sent_filtered['sentence_years'] = sent_filtered['COMMITMENT_TERM_NUM'] * sent_filtered['unit_factor']

# Override special units so they are not multiplied by COMMITMENT_TERM
# NATURAL LIFE should be a fixed 100 years regardless of COMMITMENT_TERM
sent_filtered.loc[sent_filtered['COMMITMENT_UNIT'] == 'NATURAL LIFE', 'sentence_years'] = 100
# DEATH as an extreme sentinel value (map to 150 years)
sent_filtered.loc[sent_filtered['COMMITMENT_UNIT'] == 'DEATH', 'sentence_years'] = 150
# TIME SERVED should map to 0 (or could be treated differently if needed)
sent_filtered.loc[sent_filtered['COMMITMENT_UNIT'] == 'TIME SERVED', 'sentence_years'] = 0

# print('\nSample of computed sentence_years:')
# print(sent_filtered[['COMMITMENT_TERM', 'COMMITMENT_UNIT', 'COMMITMENT_TERM_NUM', 'unit_factor', 'sentence_years']].head(20))
#Features ingineering:
print(sent_filtered['OFFENSE_CATEGORY'].value_counts())




