python main.py --c_dim 8 --dataset duke --mode train --image_dir ../data/duke/bounding_box_train --log_dir ./duke/logs \
    --model_save_dir ./duke/models --sample_dir ./duke/samples --result_dir ../data/duke/bounding_box_train_camstyle_stargan4reid

python main.py --c_dim 8 --dataset duke --mode test --image_dir ../data/duke/bounding_box_train --log_dir ./duke/logs \
    --test_iters 220000 --model_save_dir ./duke/models --sample_dir ./duke/samples --result_dir ../data/duke/bounding_box_train_camstyle_stargan4reid