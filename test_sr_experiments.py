import os
from landcover.eval_landcover_results import eval_landcover_results

TRAIN_STATE_LIST = [
    "de_1m_2013",
    "ny_1m_2013",
    "md_1m_2013",
    "pa_1m_2013",
    "va_1m_2014",
    "wv_1m_2014",
]
TEST_STATE_LIST = [
    "de_1m_2013",
    "ny_1m_2013",
    "md_1m_2013",
    "pa_1m_2013",
    "va_1m_2014",
    "wv_1m_2014",
]

GPU_IDX = 0

print("          ", end="")
for test_state in TEST_STATE_LIST:
    print(test_state, end="  ")
print("")

for train_state in TRAIN_STATE_LIST:
    print(train_state, end="")
    for test_state in TEST_STATE_LIST:
        # print("%s  %s" % (train_state, test_state))
        fn = (
            "results/results_sr_epochs_100_0/train-hr_%s_train-sr_%s/test-output_%s/log_acc_%s.txt"
            % (train_state, test_state, test_state, test_state)
        )
        if os.path.exists(fn):
            all_acc, all_jaccard, urban_acc, urban_jaccard = eval_landcover_results(fn)
            print(" " * 6 + "%0.2f" % (float(all_acc)), end="  ")
        else:
            print(" " * 10, end="  ")

    print("")
