with open('/Users/nyxinsane/Documents/Google Drive (nyXiNsane)/Credentials/Apps/Pip, Conda, and Brew/conda_packages.txt', 'r') as f:
    con = [line.rstrip(' \n') for line in f]

with open('/Users/nyxinsane/Documents/Google Drive (nyXiNsane)/Credentials/Apps/Pip, Conda, and Brew/conda_packages.txt', 'w') as f:
    for i in set([c.split('=')[0] for c in con]):
        f.write(f'{i.lower()}\n')
