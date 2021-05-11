def __get_patient_numbers():
    directories = [name for name in os.listdir(data_path) if name[0] != '.']
    return [name.split('Patient')[1] for name in directories]

def __get_patient_crises_numbers(patient):
    pass

def __save_crisis_hrv_features(patient: int, crisis: int, features: pd.DataFrame):
    assert_patient(patient)
    assert_crisis(crisis)
    try:
        file_path = '/Patient' + str(patient) + '/crisis' + str(crisis) + '_hrv_features'
        features.to_hdf(data_path + file_path, 'features', mode='a')
        print("Written in " + file_path + " was successful.")
    except IOError:
        print("That patient/crisis pair cannot be created. Save failed.")


def __read_crisis_nni(patient: int, crisis: int):
    assert_patient(patient)
    assert_crisis(crisis)
    try:
        file_path = '/Patient' + str(patient) + '/nni_Crise ' + str(crisis) + '_hospital'
        data = pd.read_hdf(data_path + file_path)
        print("Data from " + file_path + " was retreived.")
        return data
    except IOError:
        print("That patient/crisis pair does not exist. None was returned.")
        return None