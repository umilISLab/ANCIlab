pyenv activate ancilab

python3 zeroshot_classification.py --file all_data.json --model mistral --label topic_tags
python3 zeroshot_classification.py --file all_data.json --model mistral --label category_tag
python3 zeroshot_classification.py --file all_data.json --model phi3.5 --label topic_tags
python3 zeroshot_classification.py --file all_data.json --model gemma:2b --label topic_tags

python3 data_augmentation.py --file all_data.json --model mistral --label topic_tags
python3 data_augmentation.py --file all_data.json --model mistral --label category_tag

python3 feature_extraction.py --file all_data.json --model mistral --label topic_tags
python3 feature_extraction.py --file all_data.json --model mistral --label category_tag