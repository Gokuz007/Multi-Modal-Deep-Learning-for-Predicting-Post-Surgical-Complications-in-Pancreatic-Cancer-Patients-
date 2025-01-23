import os
import numpy as np
import pandas as pd
import pydicom

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
    
    return pd.DataFrame(matched_data)

def load_ct_scans_2d(clinical_file_path, raw_dir, annotated_dir, target_col):

    clinical_data = load_clinical_data(clinical_file_path, target_col)
    matched_data = match_clinical_data_with_ct(clinical_data, raw_dir, annotated_dir, target_col)

    print(f"Total records in clinical data: {len(clinical_data)}")
    print(f"Total records matched with CT scans: {len(matched_data)}")

    return matched_data
