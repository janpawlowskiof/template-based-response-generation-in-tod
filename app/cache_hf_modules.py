from app.model.pretrained_models import get_tokenizer
from app.model.pretrained_models import get_pretrained_model


def main():
    modules = [
        ("t5", "t5-large"),
        ("t5", "allegro/plt5-large"),
        ("mt5", "google/mt5-large"),
        ("t5", "google/flan-t5-large"),
        ("gpt", "ai-forever/mGPT"),
    ]
    for model_type, model_path in modules:
        print(f"downloading {model_path}")
        get_pretrained_model(model_type, model_path) 
        get_tokenizer(model_type, model_path) 


if __name__ == "__main__":
    main()
