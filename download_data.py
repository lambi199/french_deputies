import os
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

def main(output_dir='data'):
    url = 'https://www.data.gouv.fr/fr/datasets/r/092bd7bb-1543-405b-b53c-932ebb49bb8e'
    filename = 'deputes-active.csv'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
 
    output_file = os.path.join(output_dir, filename)

    if os.path.exists(output_file):
        print(f"The download file already exists in {output_file}")
    
    else:
        print("Downloading from {} ...".format(url))
        urlretrieve(url, filename=output_file)
        print("=> File saved as {}".format(output_file))


if __name__ == '__main__':
    main()