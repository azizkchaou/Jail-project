import pandas as pd
import numpy as np      
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\USER\\Desktop\\jail project\\jail.csv", 
                 encoding='latin-1', 
                 low_memory=False)
# drop rows where COMMITMENT_TERM is missing or empty (whitespace-only)
if 'COMMITMENT_TERM' in df.columns:
    empty_mask = df['COMMITMENT_TERM'].isna() | df['COMMITMENT_TERM'].astype(str).str.strip().eq('')
    n_drop = int(empty_mask.sum())
    if n_drop > 0:
        print(f"Dropping {n_drop} rows with empty COMMITMENT_TERM")
        df = df.loc[~empty_mask].reset_index(drop=True)

# drop rows where COMMITMENT_UNIT contains 'Term' or is weight/volume units
if 'COMMITMENT_UNIT' in df.columns:
    unit_str = df['COMMITMENT_UNIT'].astype(str).str.strip()
    
    # mask for 'Term' (case-insensitive)
    term_mask = df['COMMITMENT_UNIT'].astype(str).str.contains('Term', case=False, na=False)
    
    # mask for weight/volume units
    weight_units_mask = unit_str.isin(['Pounds', 'Ounces', 'Kilos', 'Grams'])
    
    # combine both conditions
    drop_mask = term_mask | weight_units_mask
    n_drop = int(drop_mask.sum())
    
    if n_drop > 0:
        print(f"Dropping {n_drop} rows where COMMITMENT_UNIT contains 'Term' or is ['Pounds', 'Ounces', 'Kilos', 'Grams']")
        df = df.loc[~drop_mask].reset_index(drop=True)

print( df.shape, )        # number of rows and columns
# df.info(),        # data types, missing values
# df.describe() ,   # stats for numeric columns
# df.head(),        # quick preview
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
# print(num_cols)
df_num=df[num_cols]
# print( df_num.shape,         # number of rows and columns
# df_num.info(),        # data types, missing values
# df_num.describe() ,   # stats for numeric columns
# df_num.head(),        # quick preview
# )

# missing value report
miss = df.isnull().sum()
miss_perc = (miss / len(df)) * 100
report = pd.DataFrame({'missing_count': miss, 'missing_perc': miss_perc})
report = report.sort_values('missing_perc', ascending=False)
print('\n=== MISSING VALUES ===')
print(report.head(50))

# drop "CHARGE_DISPOSITION_REASON" column
df = df.drop(columns=['CHARGE_DISPOSITION_REASON'])
print("Dropped column: CHARGE_DISPOSITION_REASON")

# --- add single JAIL_DAYS column (vectorized) ---
# safe references to source columns
term_raw = df['COMMITMENT_TERM'] if 'COMMITMENT_TERM' in df.columns else pd.Series(np.nan, index=df.index)
unit_raw = df['COMMITMENT_UNIT'] if 'COMMITMENT_UNIT' in df.columns else pd.Series(np.nan, index=df.index)

# parse numeric portion of COMMITMENT_TERM: try direct numeric then extract digits
term_num = pd.to_numeric(term_raw, errors='coerce')
extracted = term_raw.astype(str).str.extract(r'([+-]?\d*\.?\d+)')[0]
term_num = term_num.fillna(pd.to_numeric(extracted, errors='coerce'))

# map units to day multipliers
unit = unit_raw.astype(str).str.lower().fillna('')
conds = [
    unit.str.contains(r'year|\byr\b|yrs|\by\b', na=False),
    unit.str.contains(r'month|\bmo\b|mos', na=False),
    unit.str.contains(r'week|\bwk\b|wks', na=False),
    unit.str.contains(r'day|\bd\b|days', na=False),
    unit.str.contains(r'hour|hr', na=False),
]
choices = [365.25, 30.44, 7, 1, 1.0 / 24.0]
multipliers = np.select(conds, choices, default=np.nan)

# compute JAIL_DAYS; leave NaN when parsing fails or unit unknown
df['JAIL_DAYS'] = term_num * multipliers

# --- special case: rows where COMMITMENT_UNIT indicates Natural Life ---
# For those rows, set JAIL_DAYS = numeric(COMMITMENT_TERM) * 250
# This block is intentionally self-contained.
nl_mask = unit_raw.astype(str).str.contains(r'natural\s+life', case=False, na=False)
if nl_mask.any():
    # use term_num (already attempted numeric coercion + regex extraction)
    nl_terms = term_num[nl_mask]
    # compute replacement values (will be NaN for non-numeric terms)
    nl_days = nl_terms * 250
    df.loc[nl_mask, 'JAIL_DAYS'] = nl_days
    # print(f"Adjusted {int(nl_mask.sum())} 'Natural Life' rows: set JAIL_DAYS = COMMITMENT_TERM * 250")
    # show a small sample of affected rows for verification
    sample_cols = [c for c in ('COMMITMENT_UNIT', 'COMMITMENT_TERM', 'JAIL_DAYS') if c in df.columns]
    # print(df.loc[nl_mask, sample_cols].head(5))


#plot distribution of sentence length (in days)
# for y in num_cols:
#     plt.figure(figsize=(6, 4))
#     sns.histplot(df[y].dropna(), kde=True)
#     plt.title(f'Distribution: {y}')
#     plt.tight_layout()
#     plt.show() 

corr_matrix = df.corr(numeric_only=True)

# # print(corr_matrix)
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Correlation Heatmap of Numerical Features", fontsize=14)
# plt.show()
# print(df.head())

#convert JAIL_DAYS to integer (where not NaN):
df['JAIL_DAYS'] = df['JAIL_DAYS'].dropna().astype(int)










# save the current dataframe to CSV as requested:
out_file = 'jail_filtred.csv'
df.to_csv(out_file, index=False, encoding='utf-8')
# print(f"Saved dataframe to {out_file}")
print(df.shape)


