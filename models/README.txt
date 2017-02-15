model_gensample_v4.3
trained on 135k samples 300k iterations
---------------------------------------------------------------
test accuracy/offset RMSE/density RMSE:     0.988 / 3.805 / 0.157


model_gensample_v4.4
trained on 135k + 3x overlap dlas + low redshift slls dlas
----------------------------------------------------------------
test accuracy/offset RMSE/density RMSE:     0.988 / 3.743 / 0.154
Model saved in file: ../models/gensample_v4/model_gensample_v4.4_1540000.ckpt  - renamed to model v4.4.1


Test models
-----------
4.4.1: Trained on all data using mix test dataset
test accuracy/offset RMSE/density RMSE:     0.960 / 6.976 / 0.411
Model saved in file: ../models/gensample_v4/model_gensample_v4.4.1_475000.ckpt

4.4.2: Trained on only slls sightlines
test accuracy/offset RMSE/density RMSE:     0.964 / 7.069 / 0.497
Model saved in file: ../models/gensample_v4/model_gensample_v4.4.2_460000.ckpt

4.5.1: Training of 4.5 @ 415,000 iters using all samples after adding marked out regions


model_gensample_v5.1.0
Trained on DLAs & subDLAs for 2M iterations fully considered final

model_gensample_v5.1.1
Same as v5.1.0, just a rename because of adding bias correction to the code (no actual change to the model)

model_v6.0.0
Trained on orig_form training, just DLAs, using Model_v5
stdbuf -o 0 python localize_model.py -i 3000000 -c '../models/training/model_gensample_v6.0.0' -r '../data/gensample/orig_form/train_*' -e '../data/gensample/orig_form/test_96451.npz' | tee ../tmp/stdout_train_6.0.0.txt
stdbuf -o 0 python localize_model.py -i 3000000 -c '../models/training/model_gensample_v6.0.1' -r '../data/gensample/orig_form/train_*' -e '../data/gensample/orig_form/test_96451.npz' | tee ../tmp/stdout_train_6.0.1.txt

model_v6.1.0
stdbuf -o 0 python localize_model.py -i 3000000 -c '../models/training/model_gensample_v6.1.0' -r '../data/gensample/full_form/train_*' -e '../data/gensample/test_mix_23559.npz' | tee ../tmp/stdout_train_6.0.0.txt
Trained on full_form (double original DLAs without removals & dual dla's, + slls sightlines)
model_gensample_v6.1.1_125000	test accuracy/offset RMSE/density RMSE:     0.959 / 6.327 / 0.227

model_v6.1.2 is a copy of model_gensample_v6.1.1_125000 renamed to be the final model used in the paper
