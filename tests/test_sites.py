import numpy as np
import pandas as pd

from retina_shift.sites import (
    SITES,
    eyepacs_rows,
    idrid_rows,
    messidor_rows,
    odir_grade,
    odir_rows,
)


def test_sites_in_order():
    assert SITES == ["eyepacs", "aptos", "messidor2", "idrid", "odir"]


def test_odir_grades():
    assert odir_grade("normal fundus") == 0
    assert odir_grade("mild nonproliferative retinopathy") == 1
    assert odir_grade("moderate non proliferative retinopathy，cataract") == 2
    assert odir_grade("severe nonproliferative retinopathy") == 3
    assert odir_grade("proliferative diabetic retinopathy") == 4
    assert odir_grade("severe proliferative diabetic retinopathy") == 4
    assert odir_grade("moderate non proliferative retinopathy，severe nonproliferative retinopathy") == 3
    assert np.isnan(odir_grade("suspected diabetic retinopathy"))
    assert np.isnan(odir_grade("diabetic retinopathy"))
    assert odir_grade("suspected diabetic retinopathy，moderate non proliferative retinopathy") == 2
    assert odir_grade("glaucoma") == 0
    assert odir_grade("hypertensive retinopathy") == 0


def test_eyepacs_patient_and_eye():
    labels = pd.DataFrame({"image": ["10_left", "10_right", "13_left"], "level": [0, 2, 4]})
    members = [f"resized_train/resized_train/{i}.jpeg" for i in labels.image]
    rows = eyepacs_rows(labels, members)
    assert rows.patient_id.tolist() == ["10", "10", "13"]
    assert rows.eye.tolist() == ["left", "right", "left"]
    assert rows.referable.tolist() == [0, 1, 1]
    assert rows.member.tolist() == members


def test_messidor_keeps_gradable_and_matches_case_insensitively():
    grades = pd.DataFrame(
        {
            "image_id": ["a_PP.png", "IM001.jpg", "IM002.jpg"],
            "adjudicated_dr_grade": [0.0, 2.0, np.nan],
            "adjudicated_gradable": [1, 1, 0],
        }
    )
    members = ["messidor-2/messidor-2/preprocess/a_PP.png", "messidor-2/messidor-2/preprocess/IM001.JPG"]
    rows = messidor_rows(grades, members)
    assert rows.image_id.tolist() == ["a_PP", "IM001"]
    assert rows.member.tolist() == members
    assert rows.grade.tolist() == [0, 2]


def test_idrid_ids_are_unique_across_train_and_test():
    train = pd.DataFrame({"Image name": ["IDRiD_001"], "Retinopathy grade": [3]})
    test = pd.DataFrame({"Image name": ["IDRiD_001"], "Retinopathy grade": [0]})
    rows = idrid_rows(train, test)
    assert rows.image_id.tolist() == ["train_IDRiD_001", "test_IDRiD_001"]
    assert rows.member.str.endswith("IDRiD_001.jpg").all()
    assert rows.referable.tolist() == [1, 0]


def test_odir_one_row_per_eye_with_age_sex_and_exclusions():
    table = pd.DataFrame(
        {
            "ID": [0, 1],
            "Patient Age": [69, 57],
            "Patient Sex": ["Female", "Male"],
            "Left-Fundus": ["0_left.jpg", "1_left.jpg"],
            "Right-Fundus": ["0_right.jpg", "1_right.jpg"],
            "Left-Diagnostic Keywords": ["cataract", "suspected diabetic retinopathy"],
            "Right-Diagnostic Keywords": ["normal fundus", "moderate non proliferative retinopathy"],
        }
    )
    rows = odir_rows(table)
    assert rows.image_id.tolist() == ["0_left", "0_right", "1_right"]  # 1_left excluded (unspecified DR)
    assert rows.patient_id.tolist() == ["0", "0", "1"]
    assert rows.age.tolist() == [69, 69, 57]
    assert rows.sex.tolist() == ["female", "female", "male"]
    assert rows.referable.tolist() == [0, 0, 1]


def test_aptos_combines_its_three_splits():
    from retina_shift.sites import aptos_rows

    tables = {
        "train": pd.DataFrame({"id_code": ["a1"], "diagnosis": [2]}),
        "val": pd.DataFrame({"id_code": ["b2"], "diagnosis": [0]}),
        "test": pd.DataFrame({"id_code": ["c3"], "diagnosis": [4]}),
    }
    rows = aptos_rows(tables)
    assert rows.image_id.tolist() == ["a1", "b2", "c3"]
    assert rows.member.tolist() == [
        "train_images/train_images/a1.png",
        "val_images/val_images/b2.png",
        "test_images/test_images/c3.png",
    ]
    assert rows.referable.tolist() == [1, 0, 1]
