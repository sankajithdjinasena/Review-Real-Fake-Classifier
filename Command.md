## Colab Codes

- !python /content/drive/MyDrive/fake_reviews_project/fake_reviews_bert_finetune.py \
    --input /content/drive/MyDrive/fake_reviews_project/fake_reviews_cleaned.csv \
    --model_name distilbert-base-uncased \
    --epochs 3 \
    --batch_size 16 \
    --output_dir /content/drive/MyDrive/fake_reviews_project/bert_fake_reviews_model

- !python /content/drive/MyDrive/fake_reviews_project/fake_reviews_error_analysis.py \
    --input /content/drive/MyDrive/fake_reviews_project/fake_reviews_cleaned.csv \
    --model_dir /content/drive/MyDrive/fake_reviews_project/bert_fake_reviews_model \
    --output_csv /content/drive/MyDrive/fake_reviews_project/misclassified_reviews.csv