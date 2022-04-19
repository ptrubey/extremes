# processes.append(Popen(
#                 [sys.executable, 'test_generic.py', file, out_path, model, 
#                     '--outcome', cla_path, "--cats", str(catCols), '--sphere', 'True']
#                 ))

# python test_generic.py ./datasets/ad2_cover_x.csv ./test/results_mdppprg.pkl mdppprg --outcome ./datasets/ad2_cover_y.csv --cats [0,3] --decluster False --quantile 0.999
# python test_generic.py ./datasets/ad2_cover_x.csv ./test/results_mdpppg.pkl mdpppg --outcome ./datasets/ad2_cover_y.csv --cats [0,3] --decluster False --quantile 0.999
python test_generic.py ./test/testdata.csv ./test/results_mdppprg.pkl mdppprg --cats [3,] --decluster False --sphere True
python test_generic.py ./test/testdata.csv ./test/results_mdpppg.pkl mdpppg --cats [3,] --decluster False --sphere True
