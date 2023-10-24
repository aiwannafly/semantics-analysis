import torch

MODEL_PATH = 'cointegrated/rubert-tiny2'
HIDDEN_SIZE = 312

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PAD_TOKEN = '[PAD]'
PAD_TOKEN_IDX = 0
CLS_TOKEN_IDX = 2
SEP_TOKEN_IDX = 3
DS_PATH = 'data/formatted_term_dataset.conllu'

CLASSES = ['O', 'I-TERM']
DEFAULT_CLASS = 'O'
NUM_CLASSES = len(CLASSES)

BATCH_SIZE = 5
MAX_LENGTH = 150

EPOCHS = 40
LEARNING_RATE = 1e-5

# CLASSES = ['O',
#            'I-TERM',
#            'I-ShortName',
#            'I-Task',
#            'I-ShortName_Task',
#            'I-Value',
#            'I-Labeling',
#            'I-Date',
#            'I-Lang',
#            'I-Abbrev_Task',
#            'I-Object',
#            'I-Subject',
#            'I-Metric',
#            'I-ShortName_Metric',
#            'I-Model',
#            'I-ShortName_Model',
#            'I-Dataset',
#            'I-Organization',
#            'I-Abbrev_Organization',
#            'I-ShortName_Organization',
#            'I-Science',
#            'I-ShortName_Science',
#            'I-Person',
#            'I-Network',
#            'I-Result',
#            'I-Date_Result',
#            'I-Publication',
#            'I-Date_Method',
#            'I-Method',
#            'I-ShortName_Method',
#            'I-Application',
#            'I-Data',
#            'I-Date_Application',
#            'I-Abbrev_Application',
#            'I-Environment',
#            'I-ShortName_Environment',
#            'I-InfoResource',
#            'I-Abbrev_InfoResource',
#            'I-Activity',
#            'I-Date_Activity',
#            ]

