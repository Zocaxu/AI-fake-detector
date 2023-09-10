import os

DIR = "/vol/bitbucket/mb322/AI-fake-detection/results_test_progan-train-200k_RC"
g_results = open(os.path.join(DIR, "_results.txt"), 'w')
for subdir, dirs, files in os.walk(DIR):
    for file in files:
        f = open(os.path.join(DIR, file))
        line = f.readline()
        g_results.write(line + '\n')
        f.close()
g_results.close()
