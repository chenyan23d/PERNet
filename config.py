import os

root = './'
datasets_root = os.path.join(root,'data')

cod_training_root = os.path.join(datasets_root, 'TrainDataset')
cvc_training_root = os.path.join(datasets_root, 'TrainDataset2')
chameleon_path = os.path.join(datasets_root, 'TestDataset/CHAMELEON')
camo_path = os.path.join(datasets_root, 'TestDataset/CAMO')
cod10k_path = os.path.join(datasets_root, 'TestDataset/COD10K')
nc4k_path = os.path.join(datasets_root, 'TestDataset/NC4K')
cvc_300_path = os.path.join(datasets_root, 'TestDataset2/CVC_300')
cvc_clinicdb_path = os.path.join(datasets_root, 'TestDataset2/CVC_ClinicDB')
cvc_colondb_path = os.path.join(datasets_root, 'TestDataset2/CVC_ColonDB')
etis_path = os.path.join(datasets_root, 'TestDataset2/ETIS_LaribPolypDB')
kvasir_path = os.path.join(datasets_root, 'TestDataset2/Kvasir')
pvtv2_checkpoint_dir = './'