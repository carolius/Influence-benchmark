
python targeted_llm_manipulation/experiments/run_experiment.py --config gpt_const_veto_politics.yaml --timestamp 09_30_night
python targeted_llm_manipulation/experiments/run_experiment.py --config gpt_const_veto_action-advice.yaml --timestamp 09_30_night
bash server_utils/to_rnn_micah_transfer.sh gpt_const_veto_action-advice-09_30_night
bash server_utils/to_rnn_micah_transfer.sh gpt_const_veto_politics-09_30_night