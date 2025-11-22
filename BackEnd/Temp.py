import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import glob

def extract_features(epoch):
    features = {
        'hr_mean': np.nan,
        'hr_min': np.nan,
        'hr_max': np.nan,
        'sdnn': np.nan,
        'rmssd': np.nan,
        'acc_mean': np.nan,
        'acc_var': np.nan,
        'acc_peaks': np.nan,
        'temp_mean': np.nan,
        'temp_slope': np.nan,
        'eda_mean': np.nan,
        'eda_peaks': np.nan,
    }

    
    # HR features
    hr = epoch['HR'].dropna()
    ibi = epoch['IBI'].dropna()
    if len(hr) > 0:
        features['hr_mean'] = hr.mean()
        features['hr_min'] = hr.min()
        features['hr_max'] = hr.max()
    if len(ibi) > 1:
        features['sdnn'] = np.std(ibi)
        features['rmssd'] = np.sqrt(np.mean(np.diff(ibi)**2))
    
    # ACC features
    acc_mag = np.sqrt(epoch['ACC_X']**2 + epoch['ACC_Y']**2 + epoch['ACC_Z']**2)
    features['acc_mean'] = acc_mag.mean()
    features['acc_var'] = acc_mag.var()
    peaks, _ = find_peaks(acc_mag, height=acc_mag.mean()+acc_mag.std())
    features['acc_peaks'] = len(peaks)
    
    # TEMP features
    temp = epoch['TEMP'].dropna()
    if len(temp) > 0:
        features['temp_mean'] = temp.mean()
        features['temp_slope'] = temp.iloc[-1] - temp.iloc[0]
    
    # EDA features
    eda = epoch['EDA'].dropna()
    if len(eda) > 0:
        features['eda_mean'] = eda.mean()
        peaks, _ = find_peaks(eda, height=eda.mean()+eda.std())
        features['eda_peaks'] = len(peaks)
    
    return pd.Series(features)

# Step 1: Load all patient files
file_list = glob.glob("*_whole_df.csv")   # adjust pattern to match your filenames

dfs = []
for file in file_list:
    df_temp = pd.read_csv(file)
    # Add subject identifier from filename (e.g., "S002")
    df_temp['subject'] = file.split("_")[0]
    dfs.append(df_temp)

# Step 2: Combine into one DataFrame
df = pd.concat(dfs, ignore_index=True)

# Step 3: Epoch segmentation (30s windows)
df['epoch'] = (df['TIMESTAMP'] // 30).astype(int)

# Step 5: Apply feature extraction per subject+epoch
features = df.groupby(['subject','epoch']).apply(extract_features, include_groups=False)

# Step 6: Align features and labels
labels = df.groupby(['subject','epoch'])['Sleep_Stage'].first()
data = pd.concat([features, labels.rename('Sleep_Stage')], axis=1)
           # merge features + labels
data = data.dropna(subset=['Sleep_Stage'])  # drop rows without labels

X = data.drop(columns=['Sleep_Stage']).fillna(0)
y = data['Sleep_Stage']

# Step 7: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 8: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 9: Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))