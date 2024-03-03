import os
import pandas as pd
from sklearn.model_selection import train_test_split


remove_deputies_with_no_party ='NI'

def convert_to_days(value):

    if 'nan' in str(value):
        return value
    
    number, time = value.split(' ')
    
    if time == 'mois':
        return int(number) * 30.4167
    elif time == 'ans' or time == 'an':
        return int(number) * 365  
    else:
        return int(number)

def convert_age(value):
    if int(value) < 39 :
        return int(0)
    elif int(value) > 39:
        return int(1) 


def main(output_dir='data'):
    df_data = pd.read_csv(os.path.join(output_dir, 'deputes-active.csv'))
    df_data['experienceDepute'] = df_data['experienceDepute'].apply(convert_to_days)
    df_data['civ'] = df_data['civ'].map({'M.':1, 'Mme':0})
    df_data['age'] = df_data['age'].apply(convert_age)



    index_of_PA795100 = df_data[df_data['id'] == 'PA795100'].index
    removed_wrong_data = df_data.drop(index_of_PA795100)

    remove_deputies = removed_wrong_data[(removed_wrong_data['groupeAbrev'] == remove_deputies_with_no_party)].index
    cleaned_data_df = removed_wrong_data.drop(remove_deputies)

    df_train, df_test = train_test_split(
            cleaned_data_df, test_size=0.2, random_state=57)
    
    public_path = os.path.join('data', 'public')

    if not os.path.exists(public_path):
        os.mkdir(public_path)

    df_train.to_csv(os.path.join('data', 'train.csv'), index=False)
    df_train.to_csv(os.path.join('data', 'public', 'train.csv'), index=False)
    print('Train dataset created')
    df_test.to_csv(os.path.join('data', 'test.csv'), index=False)
    df_test.to_csv(os.path.join('data', 'public', 'test.csv'), index=False)
    print('Test dataset created')

if __name__ == '__main__':
    output_file = os.path.join('data', 'deputes-active.csv')

    if os.path.exists(output_file):
        main()
    else:
        print('You forgot to download data using python3 download_data.py')