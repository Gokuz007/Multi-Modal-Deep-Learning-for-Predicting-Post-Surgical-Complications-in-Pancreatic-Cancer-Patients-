#########################
#same as itils_new but has extra functions to load both clinical and ct scans for multi models
#########################

import os
import numpy as np
import pandas as pd
import pydicom
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def normalize(img, min_bound=-400, max_bound=400):
    """Normalize an image between "min_bound" and "max_bound", and scale between 0 and 1."""
    img = (img - min_bound) / (max_bound - min_bound)
    img[img > 1] = 0
    img[img < 0] = 0
    c = img - np.min(img)
    d = np.max(img) - np.min(img)
    img = np.divide(c, d, np.zeros_like(c), where=d != 0)
    return img

def load_raw_ct_image(file_path, min_bound=-200, max_bound=200):
    dicom = pydicom.dcmread(file_path)
    raw_image = dicom.pixel_array
    raw_image = dicom.RescaleSlope * raw_image + dicom.RescaleIntercept 
    return normalize(raw_image, min_bound, max_bound)

def load_annotated_ct_image(file_path):
    annotated_image = np.load(file_path)

    # Convert to Hounsfield Units using the given slope and intercept
    #annotated_image = annotated_image * rescale_slope + rescale_intercept

    # Normalize the image
    normalized_image = normalize(annotated_image)
    
    return normalized_image

def load_clinical_data(file_path, target_col):

    if file_path.endswith('.csv'):
        clinical_data = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        clinical_data = pd.read_excel(file_path)
    else:
        raise ValueError("File format not supported. Please use CSV or Excel.")

    # Drop rows where the target column is NaN or empty
    clinical_data = clinical_data.dropna(subset=[target_col])

    return clinical_data

def find_ct_scans_for_patient(patient_id, raw_dir, annotated_dir):
    """Find the appropriate CT scans for a patient from the raw and annotated directories."""
    if patient_id in ['130066', '130077']:
        patient_id += '_1'

    raw_files = os.listdir(raw_dir)
    annotated_files = os.listdir(annotated_dir)

    raw_image_path = None
    annotated_image_path = None

    raw_candidates = [f for f in raw_files if patient_id in f]
    if len(raw_candidates) == 1:
        raw_image_path = os.path.join(raw_dir, raw_candidates[0])
    elif len(raw_candidates) > 1:
        for f in raw_candidates:
            if '_pre' in f:
                raw_image_path = os.path.join(raw_dir, f)
                break

    annotated_candidates = [f for f in annotated_files if patient_id in f]
    if len(annotated_candidates) == 1:
        annotated_image_path = os.path.join(annotated_dir, annotated_candidates[0])
    elif len(annotated_candidates) > 1:
        for f in annotated_candidates:
            if '_pre' in f:
                annotated_image_path = os.path.join(annotated_dir, f)
                break

    return raw_image_path, annotated_image_path

def match_clinical_data_with_ct(clinical_data, raw_dir, annotated_dir, target_col):

    matched_data = []
    final_record_ids = []

    #final_record_ids.clear() 
    for _, row in clinical_data.iterrows():
        patient_id = str(row['record_id'])
        target = row[target_col]

        raw_path, annotated_path = find_ct_scans_for_patient(patient_id, raw_dir, annotated_dir)
        if raw_path and annotated_path:
            raw_image = load_raw_ct_image(raw_path)
            annotated_image = load_annotated_ct_image(annotated_path)
            stacked_image = np.stack([raw_image, annotated_image], axis=0)

            matched_data.append({
                'record_id': patient_id,
                'ct_scan': stacked_image,
                target_col: target
            })
            final_record_ids.append(patient_id)
    

    return pd.DataFrame(matched_data),final_record_ids

def load_ct_scans_2d(clinical_file_path, raw_dir, annotated_dir, target_col):

    clinical_data = load_clinical_data(clinical_file_path, target_col)
    matched_data,final_record_ids = match_clinical_data_with_ct(clinical_data, raw_dir, annotated_dir, target_col)

    print(f"Total records in clinical data: {len(clinical_data)}")
    print(f"Total records matched with CT scans: {len(matched_data)}")

    return matched_data, final_record_ids

def select_cols(clinical_data):

    columns_to_retain = [
    'record_id',
    'gender_castor_x',
    'age_surgery_x',
    'datopn',
    'neoadjuvant_castor_10',
    'sandostatine',
    'typok',
    'drain_castor',
    'sof',
    'mof',
    'operative_bloodloss_compl',
    'pa_groups',
    'lengte',
    'gewicht',
    #'gewverlies',
    'bilirubine',
    'hemoglobine',
    #'albu',
    'origine',
    'histdiagnpost',
    #'diameterpost',
    #'radmarge',
    #'differentiatie',
    #'diffnet',
    #'stadptpanc2018',
    #'crp',
    'smra',
    'muscle_area',
    'vat_area',
    'sat_area',
    #'smra_neo',

    'drain_aantal',
    'operative_time',
    'aantal_invasive_intervent',
    'ic_uopname',
    'adjuvant_treatment',
    'pd',
    'invasive_interv',
    'ic',
    'ic_later',
    'neoadjuvant_yn',
    'crpopf',
    'c_rgallek',
    'c_rchylus',
    'crdge',
    'crpph',
    'minimally_invasive_resection_yn',
    'invasief',
    'resecveneus',
    'resecart',
    'resecaanv',
    'drain_dpca',
    'octreo',
    'compl',
    'pneumonie',
    'wondinfectie',
    'transfusie',

    #'neoadjuvant_chemo',
    #'implementphase',
    #'typechemotherapy',
    #'minimally_invasive_resection_yn',
    #'stadptpanc',
    'radicaliteit',
    #'major_complications'
    ]

    df = clinical_data[columns_to_retain]

    return df

def preprocess_clinical_data(clinical_data):
    df = clinical_data
    df = df.copy()
    #Date Feature
    #replace missing data with place holder
    df['datopn'] = pd.to_datetime(df['datopn'],errors = 'coerce')
    
    df['datopn'] = df['datopn'].fillna(pd.to_datetime('2018-12-11 00:00:00'))
    
    df['dat_year'] = df['datopn'].dt.year
    df['dat_month'] = df['datopn'].dt.month
    df.drop('datopn',axis=1,inplace =True)

    #numerical features
    numerical_feraures = [
        'age_surgery_x',
        'operative_bloodloss_compl',
        'lengte',
        'gewicht',
        'bilirubine',
        'hemoglobine',
        'smra',
        'muscle_area',
        'vat_area',
        'sat_area',

        'operative_time'
    ]

    #Initialize imputer with median
    imputer_num = SimpleImputer(strategy = 'median')

    #transform the numerical features
    df[numerical_feraures] = imputer_num.fit_transform(df[numerical_feraures])

    #categ features
    categorical_features = [
        'gender_castor_x',
        'neoadjuvant_castor_10',
        'sandostatine',
        'typok',
        'drain_castor',
        'sof',
        'mof',
        'origine',
        'histdiagnpost',
        'pa_groups',

        'drain_aantal',
        'aantal_invasive_intervent',
        'ic_uopname',
        'adjuvant_treatment',
        'pd',
        'invasive_interv',
        'ic',
        'ic_later',
        'neoadjuvant_yn',
        'crpopf',
        'c_rgallek',
        'c_rchylus',
        'crdge',
        'crpph',
        'minimally_invasive_resection_yn',
        'invasief',
        'resecveneus',
        'resecart',
        'resecaanv',
        'drain_dpca',
        'octreo',
        #'compl',
        'pneumonie',
        'wondinfectie',
        'transfusie',
        #'implementphase',
        #'minimally_invasive_resection_yn',
        'radicaliteit'
    ]

    # Initialize the imputer with mode
    imputer_cat = SimpleImputer(strategy='most_frequent')

    df[categorical_features] = imputer_cat.fit_transform(df[categorical_features])

    #selecting relevaant columns after chi2 test and correlation 
    df = df[['record_id','gender_castor_x','sandostatine','typok',
         #'sof','mof',
         'operative_bloodloss_compl',
         'muscle_area','gewicht','vat_area',
         'bilirubine','hemoglobine', 'age_surgery_x',
         'origine','pa_groups','operative_time',
         #'aantal_invasive_intervent',
         'ic_uopname', 'adjuvant_treatment', 'pd',
         #'invasive_interv',
         #'ic', 'ic_later',
         #'crpopf', 'c_rgallek', 'c_rchylus', 'crdge', 'crpph',
         'invasief',
         'drain_dpca',
         'octreo',
         'compl',
         #'pneumonie', 'wondinfectie', 'transfusie',

        ]]

    
    #Encoding Categorical Variables
    binary_features = ['gender_castor_x',
                   #'sof', 'mof',
                   'ic_uopname','adjuvant_treatment', 'pd',
                   #'invasive_interv',
                   #'ic', 'ic_later',
                   #'crpopf', 'c_rgallek', 'c_rchylus', 'crdge', 'crpph',
                   'drain_dpca',
                   'octreo'
                   #'compl', 'pneumonie', 'wondinfectie', 'transfusie'
                  # 'radicaliteit'
                  ]

    df[binary_features] = df[binary_features].astype(int)

    #multi-class categorical features to be one-hot encoded
    multi_class_features = [
    #'neoadjuvant_castor_10',
    'sandostatine',
    'typok',
    'pa_groups',
    'origine',
    #'histdiagnpost'
    'invasief'
    ]

    df = pd.get_dummies(df, columns = multi_class_features, drop_first=False)

    #Feature Scaling
    numerical_features_scaled = [
    'age_surgery_x',
    'operative_bloodloss_compl',
    #'lengte',
    'gewicht',
    'bilirubine',
    'hemoglobine',
    #'smra',
    'muscle_area',
    'vat_area',
    #'sat_area',
    #'dat_year',
    #'dat_month'
    'operative_time',
    #'aantal_invasive_intervent'
    ]

    #intitalize scaler
    scaler = StandardScaler()
    df[numerical_features_scaled] = scaler.fit_transform(df[numerical_features_scaled])

    df = df.dropna(subset=['compl']) 

    

    return df
    

def load_and_filter_clinical_data(clinical_file_path, target_col, final_record_ids):
    #print(f"Number of patients with CT scans detected: {len(final_record_ids)}")
    clinical_data = load_clinical_data(clinical_file_path, target_col)

    clinical_data['record_id'] = clinical_data['record_id'].astype(str)
    final_record_ids = list(map(str, final_record_ids))

    # Filter clinical_data based on final_record_ids
    clinical_data = clinical_data[clinical_data['record_id'].isin(final_record_ids)]
    #print(f"Filtered clinical data records: {len(clinical_data)} (should match final_record_ids count: {len(final_record_ids)})")

    if clinical_data.empty:
        print("Warning: No matching records found in clinical data for final_record_ids.")
        return pd.DataFrame()  # Return empty DataFrame if no matches are found

    clinical_data = select_cols(clinical_data)
    df = preprocess_clinical_data(clinical_data)
    return df


