# generate_review.py

from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the fine-tuned model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("./finetuned_codet5")
model = T5ForConditionalGeneration.from_pretrained("./finetuned_codet5")

def generate_review(code_snippet):
    input_text = "review: " + code_snippet
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    review = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return review

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python generate_review.py '<code_snippet>'")
        sys.exit(1)

    code_snippet = sys.argv[1]
    review = generate_review(code_snippet)
    print("Review:", review)
