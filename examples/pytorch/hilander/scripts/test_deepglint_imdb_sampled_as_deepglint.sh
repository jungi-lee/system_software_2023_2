python test_subg.py --data_path data/subcenter_arcface_deepglint_imdb_features_sampled_as_deepglint_1_in_10.pkl --model_filename checkpoint/deepglint_sampler.pth --knn_k 10 --tau 0.8 --level 10 --threshold prob --faiss_gpu --hidden 512 --num_conv 1 --batch_size 4096 --early_stop --use_cluster_feat
