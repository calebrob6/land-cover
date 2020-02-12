TEST_SPLITS=[
    "de_1m_2013"
    "ny_1m_2013"
    "md_1m_2013"
    "pa_1m_2013"
    "va_1m_2014"
    "wv_1m_2014"
]
EXP_BASE="/results/results_sr_epochs_100_0/train-hr_%s_train-sr_%s/"

def eval_landcover_results(acc_file):
    with open(acc_file,"r") as src:
        lines = src.readlines()
        accuracy_jaccard_all(lines)
        accuracy_jaccard_nlcd(lines)

def accuracy_jaccard_all(lines):
    lines_8 = lines[-8:0]
    lines_4 = lines_8[0:4]
    lines_processed = [process_line(line) for line in line4]
    print()

def accuracy_jaccard_nlcd(lines):
    print()

def process_line(line):

    return result_list

def eval_all_landcover_results(test_splits=TEST_SPLITS):
    for split in test_splits:
        eval_landcover_results(EXP_BASE % (split,split) + "/test-output_%s/log_acc_%s.txt" % (split,split))

eval_all_landcover_results()
